import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLR_Module(nn.Module):
    def __init__(self, channels, sigma_sp=2.0, eps=1e-8):
        super().__init__()
        self.sigma_sp = sigma_sp
        self.eps = eps

        # W: learnable projection W(F_j)
        self.W = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # 4-connected에서는 ||x_i - x_j|| = 1 이라 spatial term은 상수 하나
        # exp(-1 / (2 * sigma^2)) 값 미리 buffer로 저장
        spatial_gauss = math.exp(-1.0 / (2.0 * (sigma_sp ** 2)))
        self.register_buffer(
            "spatial_gauss_4",
            torch.tensor(spatial_gauss, dtype=torch.float32)
        )

    def forward(self, features, panoptic_mask):
        """
        features:      [B, C, H, W]
        panoptic_mask: [B, H_orig, W_orig]  (long, instance IDs, sky=0은 이미 제거된 상태라고 가정)
        """
        B, C, H, W = features.shape

        # 1. panoptic을 feature 해상도에 맞게 nearest로 다운샘플
        inst_mask = F.interpolate(
            panoptic_mask.unsqueeze(1).float(),  # [B,1,H_orig,W_orig]
            size=(H, W),
            mode="nearest"
        ).squeeze(1).long()                     # [B, H, W]

        # 2. 4-connected inst affinity 계산 (α_ij)
        #    shape: [B, H, W, 4]  (4 방향: right, left, down, up)
        alpha = self.compute_affinity_4connected(inst_mask)

        # 3. 정규화: α_hat_ij = α_ij / (∑_k α_ik + eps)
        alpha_hat = alpha / (alpha.sum(dim=-1, keepdim=True) + self.eps)  # [B,H,W,4]

        # 4. message passing: ∑_j α_hat_ij · W(F_j)
        F_transformed = self.W(features)  # [B,C,H,W]
        F_tilde = self.message_passing_4connected(F_transformed, alpha_hat)  # [B,C,H,W]

        # 5. residual
        F_out = features + F_tilde
        return F_out

    def compute_affinity_4connected(self, inst_mask):
        """
        inst_mask: [B, H, W] (instance ID map)
        returns:   [B, H, W, 4]  (4 방향에 대한 α_ij)
        """
        B, H, W = inst_mask.shape
        device = inst_mask.device

        alpha = torch.zeros(B, H, W, 4, device=device, dtype=torch.float32)

        # 4 directions: (dy, dx)
        # roll(shifts=(dy,dx)) -> neighbor index는 (y-dy, x-dx)
        # 여기서는 방향 이름이 중요하진 않고, 4-connected graph만 맞으면 됨
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up (개념상)

        for d, (dy, dx) in enumerate(shifts):
            # Shifted instance mask: inst_shifted(y,x) = inst_mask(y-dy, x-dx)
            inst_shifted = torch.roll(inst_mask, shifts=(dy, dx), dims=(1, 2))

            # 1[inst(i) == inst(j)]
            inst_match = (inst_mask == inst_shifted).float()  # [B,H,W]

            # Spatial Gaussian term: exp(-1 / (2 * sigma^2)) (상수)
            alpha_d = inst_match * self.spatial_gauss_4  # [B,H,W]

            # 경계에서 wrap-around 막기 (roll 때문에 생기는 가짜 이웃 제거)
            if dx == 1:      # neighbor index는 (x-1) → x=0이면 이웃 없음
                alpha_d[:, :, 0] = 0.0
            elif dx == -1:   # neighbor index는 (x+1) → x=W-1이면 이웃 없음
                alpha_d[:, :, -1] = 0.0
            elif dy == 1:    # neighbor index는 (y-1) → y=0이면 이웃 없음
                alpha_d[:, 0, :] = 0.0
            elif dy == -1:   # neighbor index는 (y+1) → y=H-1이면 이웃 없음
                alpha_d[:, -1, :] = 0.0

            alpha[:, :, :, d] = alpha_d

        return alpha

    def message_passing_4connected(self, features, alpha_hat):
        """
        features:  [B, C, H, W]   (이미 W(F) 적용된 feature)
        alpha_hat: [B, H, W, 4]
        returns:   [B, C, H, W]
        """
        B, C, H, W = features.shape
        device = features.device

        F_tilde = torch.zeros_like(features, device=device)

        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # compute_affinity와 동일 순서

        for d, (dy, dx) in enumerate(shifts):
            # F_shifted(y,x) = F(y-dy, x-dx) = neighbor j의 feature
            F_shifted = torch.roll(features, shifts=(dy, dx), dims=(2, 3))  # [B,C,H,W]

            # α_hat_ij를 [B,1,H,W]로 만들어서 채널에 브로드캐스트
            weight = alpha_hat[:, :, :, d].unsqueeze(1)  # [B,1,H,W]

            # 가중 합: ∑_j α_hat_ij · F_j
            F_tilde += weight * F_shifted

        return F_tilde
