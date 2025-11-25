import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Union
import numpy as np
import os
import cv2

from FLR_Module import FLR_Module


class conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        has_bn=True, 
        has_relu=True,
        **kwargs
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = nnf.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class resnext_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, has_proj=False):
        super().__init__()
        bottleneck = out_channels//4
        assert (bottleneck % groups == 0) and (bottleneck / groups) % 4 == 0, (bottleneck, groups)
        self.conv_1x1_shrink = conv(in_channels, bottleneck, kernel_size=1, padding=0)
        self.conv_3x3        = conv(bottleneck,  bottleneck, kernel_size=3, stride=stride, groups=groups)
        self.conv_1x1_expand = conv(bottleneck,  out_channels, kernel_size=1, padding=0, has_relu=False) 

        self.has_proj = has_proj
        if self.has_proj:
            if stride == 2:
                self.dsp = nn.AvgPool2d(kernel_size=2, stride=2)
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, padding=0, has_relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = x
        if self.has_proj:
            if hasattr(self, "dsp"):
                proj = self.dsp(proj)
            proj = self.shortcut(proj)
        x = self.conv_1x1_shrink(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1_expand(x)
        x = x + proj
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, base_ch=16):
        super().__init__()

        # full > 1/2 (conv 1)
        self.conv00 = conv(in_channels, base_ch, stride=2) # 채널 3 > 16

        # 1/2 > 1/4 (Layer 1)
        self.res1_0 = resnext_block(base_ch, base_ch*2, stride=2, groups=2, has_proj=True) # 채널 16 > 32
        self.res1_1 = resnext_block(base_ch*2, base_ch*2, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        self.res1_2 = resnext_block(base_ch*2, base_ch*2, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        
        # 1/4 > 1/8 (Layer 2)
        self.res2_0 = resnext_block(base_ch*2, base_ch*4, stride=2, groups=2, has_proj=True) # 채널 32 > 64
        self.res2_1 = resnext_block(base_ch*4, base_ch*4, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        self.res2_2 = resnext_block(base_ch*4, base_ch*4, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가

        # 1/8 > 1/16 (Layer 3)
        self.res3_0 = resnext_block(base_ch*4, base_ch*8, stride=2, groups=2, has_proj=True) # 채널 64 > 128
        self.res3_1 = resnext_block(base_ch*8, base_ch*8, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        self.res3_2 = resnext_block(base_ch*8, base_ch*8, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가

        # 1/16 > 1/32 (Layer 4)
        self.res4_0 = resnext_block(base_ch*8, base_ch*16, stride=2, groups=2, has_proj=True) # 채널 128 > 256
        self.res4_1 = resnext_block(base_ch*16, base_ch*16, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        self.res4_2 = resnext_block(base_ch*16, base_ch*16, stride=1, groups=2, has_proj=False) # 해상도, 채널 유지 및 표현력 증가
        
    def forward(self,x):
        # full > 1/2
        x0 = self.conv00(x)

        # 1/2 > 1/4 
        x1= self.res1_0(x0)
        x1= self.res1_1(x1)
        x1= self.res1_2(x1)

        # 1/4 > 1/8
        x2= self.res2_0(x1)
        x2= self.res2_1(x2)
        x2= self.res2_2(x2)

        # 1/8 > 1/16
        x3= self.res3_0(x2)
        x3= self.res3_1(x3)
        x3= self.res3_2(x3)

        # 1/16 > 1/32
        x4= self.res4_0(x3)
        x4= self.res4_1(x4)
        x4= self.res4_2(x4)

        return x0, x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self, base_ch=16, num_features=256, max_depth: float = 80.0):
        super().__init__()

        self.max_depth = max_depth

        H_2 = base_ch # 16
        H_4 = base_ch*2 # 32
        H_8 = base_ch*4 # 64
        H_16 = base_ch*8 # 128
        H_32 = base_ch*16 # 256

        # 1/32 > 1/16
        self.upconv5 = upconv(H_32, num_features) # 채널 256 > 256 
        self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + H_16, num_features, 3, 1, 1, bias=False),
                                              nn.ELU())

        # 1/16 > 1/8
        self.upconv4    = upconv(num_features, num_features // 2) # 채널 256 > 128
        self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + H_8, num_features // 2, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        # 1/8 > 1/4
        self.upconv3    = upconv(num_features // 2, num_features // 4) # 채널 128 > 64
        self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + H_4, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())

        # 1/4 > 1/2
        self.upconv2    = upconv(num_features // 4, num_features // 8) # 채널 64 > 32
        self.bn2        = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features // 8 + H_2, num_features // 8, 3, 1, 1, bias=False),
                                              nn.ELU())

        # 1/2 > Full
        self.upconv1    = upconv(num_features // 8, num_features // 16) # 채널 32 > 16
        self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 3, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())

        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

        self.use_seg = True   # 일단 켜두고, 나중에 args로 빼도 됨
        self.num_seg_classes = 8  # coarse 8-class

        if self.use_seg:
            # 1) H/4 → H/2
            self.seg_upconv2 = upconv(num_features // 4, num_features // 8)
            self.seg_bn2     = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
            self.seg_conv2   = torch.nn.Sequential(nn.Conv2d(num_features // 8 + H_4, num_features // 8, 3, 1, 1, bias=False),
                                                    nn.ELU())

            # 2) H/2 → H
            self.seg_upconv1 = upconv(num_features // 8, num_features // 16)
            self.seg_conv1   = torch.nn.Sequential(nn.Conv2d(num_features // 16 + H_2, num_features // 16, 3, 1, 1, bias=False),
                                                    nn.ELU())

            # 3) 최종 seg head
            self.seg_head    = nn.Conv2d(
                num_features // 16,
                self.num_seg_classes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        
    def forward(self, x0, x1, x2, x3, x4, x_img):
        # x4: H/32, x3: H/16, x2: H/8, x1: H/4, x0: H/2, x_img: H

        # 1) H/32 → H/16
        u5 = self.upconv5(x4)          # [B, 256, H/16, W/16]
        u5 = self.bn5(u5)
        c5 = torch.cat([u5, x3], dim=1)   # [B, 256+128, H/16, W/16]
        f5 = self.conv5(c5)               # 256+128 에서 256으로 채널 맞춰주기

        # 2) H/16 → H/8
        u4 = self.upconv4(f5)          # [B, 128, H/8, W/8]
        u4 = self.bn4(u4)
        c4 = torch.cat([u4, x2], dim=1)   # [B, 128+64, H/8, W/8]
        f4 = self.conv4(c4)               # 128+64에서 128로 채널 맞춰주기
        f4 = self.bn4_2(f4)

        # 3) H/8 → H/4
        u3 = self.upconv3(f4)          # [B, 64, H/4, W/4]
        u3 = self.bn3(u3)
        c3 = torch.cat([u3, x1], dim=1)   # [B, 64+32, H/4, W/4]
        f3 = self.conv3(c3)               # 64+32에서 64로 채널 맞춰주기
        # H/4 피처 받아서 H/2로 업샘플 (seg 브랜치용)
        if self.use_seg:
            seg_u2 = self.seg_upconv2(f3)      # [B, Cseg, H/2, W/2]
            seg_u2 = self.seg_bn2(seg_u2)      

        # 4) H/4 → H/2
        u2 = self.upconv2(f3)          # [B, 32, H/2, W/2]
        u2 = self.bn2(u2)
        c2 = torch.cat([u2, x0], dim=1)   # [B, 32+16, H/2, W/2]
        f2 = self.conv2(c2)               # [B, 32, H/2, W/2]

        # depth f2랑 업샘플한 seg2 concat 후 H/2 피처 받아서 H(full)로 업샘플
        if self.use_seg:
            f2_cat = torch.cat([f2, seg_u2], dim=1) # [B, 32+32, H/2, W/2]
            f2 = self.seg_conv2(f2_cat)  # 32+32에서 32로 맞춰주기

            seg_u1 = self.seg_upconv1(f2) 

        # 5) H/2 → H
        u1 = self.upconv1(f2)          # [B, 16, H, W]

        assert x_img.shape[2:] == u1.shape[2:], \
            f"Shape mismatch: x_img={x_img.shape[2:]}, u1={u1.shape[2:]}"

        c1 = torch.cat([u1, x_img], dim=1)   # [B, 16+3, H, W]
        f1 = self.conv1(c1)                  # [B, 16, H, W]

        #depth f1이랑 seg1 concat
        if self.use_seg:
            f1_cat = torch.cat([f1, seg_u1], dim=1)   # [B, 16 + 16, H/2, W/2]
            f1 = self.seg_conv1(f1_cat)               # 16+16에서 16으로 맞춰주기
            seg_logits = self.seg_head(f1) 

        depth_norm = self.get_depth(f1)      # [B, 1, H, W], 0~1
        depth = self.max_depth * depth_norm  # [B, 1, H, W], 0~max_depth

        return depth, seg_logits



class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(in_channels=3, base_ch=16)
        self.decoder = Decoder(base_ch=16, num_features=256, max_depth=80.0)

        # 🔥 FLR teacher modules (feature-level regularizer)
        H_2 = 16
        H_4 = 32
        H_8 = 64
        H_16 = 128
        H_32 = 256 

        self.use_flr = True
        self.flr_1_2 = FLR_Module(H_2)
        self.flr_1_4  = FLR_Module(H_4)
        self.flr_1_8  = FLR_Module(H_8)
        self.flr_1_16 = FLR_Module(H_16)
        self.flr_1_32 = FLR_Module(H_32)

    def compute_llr_loss(self, pred, teacher, pp_inst, pp_valid_mask):
        """
        pred        : [B, 1, H, W]  (student depth)
        teacher     : [B, 1, H, W]  (teacher depth, DepthPro)
        pp_inst     : [B, H, W] or [B, 1, H, W]  (instance id map)
        pp_valid_mask : [B, 1, H, W]  (panoptic valid mask, 0/1)

        역할:
        - 같은 instance 안에서 3x3 이웃 픽셀 쌍 (i,j)에 대해
          Δs = D_s(i) - D_s(j)
          Δt = D_t(i) - D_t(j)
          의 차이를 Smooth L1로 맞추는 loss.
        """

        # pp_inst shape 정리: [B,H,W]
        if pp_inst.dim() == 4:
            assert pp_inst.size(1) == 1, f"pp_inst must be [B,1,H,W] or [B,H,W], got {pp_inst.shape}"
            inst = pp_inst[:, 0]  # [B,H,W]
        elif pp_inst.dim() == 3:
            inst = pp_inst
        else:
            raise ValueError(f"pp_inst must be [B,H,W] or [B,1,H,W], got {pp_inst.shape}")

        B, _, H, W = pred.shape

        # [B,1,H,W] -> [B,H,W]
        D_s = pred[:, 0]
        D_t = teacher[:, 0]
        valid_pan = (pp_valid_mask[:, 0] > 0.5)  # [B,H,W]

        # teacher depth가 0 이상인 곳만 사용 (클리핑 범위 내)
        teacher_valid = (D_t > 0.0)
        # 최종 valid mask: panoptic valid ∩ teacher 유효
        valid = valid_pan & teacher_valid  # [B,H,W]

        # 아무 것도 없으면 0 리턴
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        # 8-connected 방향 (3x3 이웃)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        total_loss = pred.new_tensor(0.0)
        total_count = pred.new_tensor(0.0)

        for dy, dx in directions:
            # roll로 neighbor 가져오기
            D_s_n = torch.roll(D_s, shifts=(dy, dx), dims=(1, 2))
            D_t_n = torch.roll(D_t, shifts=(dy, dx), dims=(1, 2))
            inst_n = torch.roll(inst, shifts=(dy, dx), dims=(1, 2))
            valid_n = torch.roll(valid, shifts=(dy, dx), dims=(1, 2))

            # 같은 instance & instance id != 0
            same_inst = (inst == inst_n) & (inst != 0)
            # center / neighbor 둘 다 valid
            both_valid = valid & valid_n
            # teacher neighbor도 유효
            teacher_valid_n = (D_t_n > 0.0)

            m = same_inst & both_valid & teacher_valid_n  # [B,H,W], bool

            # 경계 wrap-around 제거 (roll로 생기는 엣지 가짜 이웃 제거)
            border = torch.ones_like(m, dtype=torch.bool)

            # 세로 방향 경계 제거
            if dy > 0:
                # 위쪽 dy줄이 roll로 wrap된 부분 → 제거
                border[:, :dy, :] = False
            elif dy < 0:
                # 아래쪽 |dy|줄이 wrap된 부분 → 제거
                border[:, H+dy:, :] = False  # dy는 음수

            # 가로 방향 경계 제거
            if dx > 0:
                # 왼쪽 dx열이 wrap된 부분 → 제거
                border[:, :, :dx] = False
            elif dx < 0:
                # 오른쪽 |dx|열이 wrap된 부분 → 제거
                border[:, :, W+dx:] = False  # dx는 음수

            m = m & border


            if m.sum() == 0:
                continue

            # gradient (차분)
            delta_s = D_s - D_s_n  # [B,H,W]
            delta_t = D_t - D_t_n  # [B,H,W]

            diff = (delta_s - delta_t)[m]  # [N]
            if diff.numel() == 0:
                continue

            abs_diff = diff.abs()
            smooth_mask = (abs_diff < 1.0).float()
            value = 0.5 * (diff ** 2) * smooth_mask + (abs_diff - 0.5) * (1.0 - smooth_mask)

            total_loss += value.sum()
            total_count += diff.numel()

        if total_count < 1.0:
            return pred.new_tensor(0.0)

        return total_loss / total_count
    # ===== LLR 메서드 끝 =====

    def forward(self, img):
        x0, x1, x2, x3, x4 = self.encoder(img)
        depth, seg_logits = self.decoder(x0, x1, x2, x3, x4, img)
        return depth, seg_logits

    def forward_with_feats(self, img):
        x0, x1, x2, x3, x4 = self.encoder(img)
        depth, seg_logits = self.decoder(x0, x1, x2, x3, x4, img)
        return depth, seg_logits, (x0, x1, x2, x3, x4)

    @staticmethod
    def make_torch_tensor(data:Union[np.ndarray, torch.Tensor], device, dtype) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint16:
                data = data.astype(np.float32)
            data = torch.from_numpy(data)
        if data.device != device:
            data = data.to(device)
        if data.dtype != dtype:
            data = data.to(dtype)
        return data

    @staticmethod
    def padding(data, divisor=32, dim=0):
        # dim: 0 for 2 axis, 1 for height, 2 for width
        if dim == 0:
            B, C, H, W = data.shape
            new_h, new_w = ((H + divisor - 1) // divisor) * divisor, ((W + divisor - 1) // divisor) * divisor
            new_data = torch.zeros((B, C, new_h, new_w), device=data.device, dtype=data.dtype)
            new_data[:, :, :H, :W] = data
        elif dim == 1:
            B, C, H, W = data.shape
            new_h, new_w = ((H + divisor - 1) // divisor) * divisor, ((W + divisor - 1) // divisor) * divisor
            new_data = torch.zeros((B, C, new_h, W), device=data.device, dtype=data.dtype)
            new_data[:, :, :H, :W] = data
        else:
            B, C, H, W = data.shape
            new_h, new_w = ((H + divisor - 1) // divisor) * divisor, ((W + divisor - 1) // divisor) * divisor
            new_data = torch.zeros((B, C, H, new_w), device=data.device, dtype=data.dtype)
            new_data[:, :, :H, :W] = data
        return new_data

    @property
    def device(self, ):
        device_set = set([p.device for p in self.parameters()])
        assert len(device_set) == 1
        return device_set.pop()

    @property
    def dtype(self, ):
        dtype_set = set([p.dtype for p in self.parameters()])
        assert len(dtype_set) == 1
        return dtype_set.pop() 


    def downsample_depthmap(self, inp, factor):
        assert factor<=1
        n,c,h,w = inp.shape
        new_h, new_w = int(h*factor), int(w*factor)
        out = torch.zeros((n,c,new_h,new_w), device=self.device)

        nonzero_idx = list(torch.nonzero(inp, as_tuple=True))
        nonzero_arr = inp[nonzero_idx]
        nonzero_idx[2] = (nonzero_idx[2]*factor).long()
        nonzero_idx[3] = (nonzero_idx[3]*factor).long()
        out[nonzero_idx] = nonzero_arr
        return out

    @staticmethod
    def get_loss_l1_smooth(pred, label, mask):
        pred        = pred.reshape(pred.shape[0], -1)
        label       = label.reshape(label.shape[0], -1)
        mask        = mask.reshape(mask.shape[0], -1)
        diff        = (pred - label) * mask
        smooth_mask = (diff.abs() < 1.0).float()
        value       = (0.5 * diff ** 2) * smooth_mask + (diff.abs() - 0.5) * (1.0 - smooth_mask)
        serr        = value.sum(axis=1).mean() / mask.sum()
        L           = 1.0  #  label.partial_shape[1]
        loss        = serr / L
        return loss

    


    def forward_train(self, mini_batch_data: dict):
        img = self.padding(self.make_torch_tensor(mini_batch_data["img"], self.device, self.dtype))
        label = self.padding(self.make_torch_tensor(mini_batch_data["label"], self.device, self.dtype))
        label_mask = self.padding(self.make_torch_tensor(mini_batch_data['label_mask'], self.device, self.dtype))
        pp_inst = self.padding(self.make_torch_tensor(mini_batch_data['pp_inst'], self.device, torch.long))
        pp_sem = self.padding(self.make_torch_tensor(mini_batch_data['pp_sem'], self.device, torch.long))
        pp_valid_mask = self.padding(self.make_torch_tensor(mini_batch_data['pp_valid_mask'], self.device, self.dtype))
        teacher = self.padding(self.make_torch_tensor(mini_batch_data["teacher_depth"], self.device, self.dtype))

        pred, seg_logits, (x0, x1, x2, x3, x4) = self.forward_with_feats(img)

        l1_loss = self.get_loss_l1_smooth(pred, label, label_mask)
        flr_loss = pred.new_tensor(0.0)
        distill_loss = pred.new_tensor(0.0)
        llr_loss = pred.new_tensor(0.0)
        seg_loss = pred.new_tensor(0.0)

        llr_valid_mask = (teacher > 0.0).float() * pp_valid_mask  # [B,1,H,W]
        llr_loss = self.compute_llr_loss(
                pred,          # student depth [B,1,H,W]
                teacher,       # teacher depth [B,1,H,W]
                pp_inst,       # [B,1,H,W] or [B,H,W]
                llr_valid_mask # [B,1,H,W]
            )

        # teacher가 0 이상인 픽셀만 사용 (0이면 없는 depth로 가정)
        teacher_valid = (teacher > 0.0).float()
        distill_loss = self.get_loss_l1_smooth(pred, teacher, teacher_valid)
        if (seg_logits is not None) and self.decoder.use_seg:
            # pp_sem: [B,1,H,W] or [B,H,W] -> [B,H,W]
            if pp_sem.dim() == 4:
                pp_sem_flat = pp_sem[:, 0, :, :]   # [B,H,W]
            else:
                pp_sem_flat = pp_sem               # [B,H,W]

            pp_sem_flat = pp_sem_flat.long()

            # valid mask: [B,1,H,W] -> [B,H,W]
            valid_seg = (pp_valid_mask[:, 0, :, :] > 0.5).float()  # [B,H,W]

            # CE loss per pixel: [B,H,W]
            ce_per_pixel = nnf.cross_entropy(
                seg_logits,          # [B,C,H,W], C = 8
                pp_sem_flat,         # [B,H,W], 값: {0~5, 255}
                reduction="none",
                ignore_index=255,
            )

            # valid 영역만 평균
            seg_loss = (ce_per_pixel * valid_seg).sum() / (valid_seg.sum() + 1e-6)

        # if self.use_flr:
        #     # pp_inst 형태 정리: [B,H,W] 또는 [B,1,H,W] 둘 다 허용
        #     if pp_inst.dim() == 4:
        #         assert pp_inst.size(1) == 1, f"pp_inst must be [B,1,H,W] or [B,H,W], got {pp_inst.shape}"
        #         pp_inst_feat = pp_inst[:, 0]          # [B,H,W]
        #     elif pp_inst.dim() == 3:
        #         pp_inst_feat = pp_inst                 # [B,H,W]
        #     else:
        #         raise ValueError(f"pp_inst must be [B,H,W] or [B,1,H,W], got {pp_inst.shape}")

        #     # 라이다 유효 + 파놉틱 유효 둘 다 만족하는 영역만 사용
        #     valid = pp_valid_mask.detach()     # [B,1,H,W]
            

        #     # 각 scale에 맞게 valid mask downsample
        #     # valid_1_2 = nnf.interpolate(valid, size=x0.shape[2:], mode='nearest')  # x0: 1/2
        #     #valid_1_4  = nnf.interpolate(valid, size=x1.shape[2:], mode='nearest')  # x1: 1/4
        #     #valid_1_8  = nnf.interpolate(valid, size=x2.shape[2:], mode='nearest')  # x2: 1/8
        #     #valid_1_16 = nnf.interpolate(valid, size=x3.shape[2:], mode='nearest')  # x3: 1/16
        #     #valid_1_32 = nnf.interpolate(valid, size=x4.shape[2:], mode='nearest')


        #     # 🔹 Teacher feature: FLR_Module(features_detach, pp_inst)
        #     #    - feature는 detach: student만 teacher를 따라가게
        #     #    - FLR_Module 파라미터는 학습됨 (detach 안 함)
        #     #x0_T = self.flr_1_2(x0.detach(), pp_inst_feat)  # [B,16,H/2,W/2]
        #     #x1_T = self.flr_1_4(x1.detach(),  pp_inst_feat)   # [B,32,H/4,W/4]
        #     #x2_T = self.flr_1_8(x2.detach(),  pp_inst_feat)   # [B,64,H/8,W/8]
        #     #x3_T = self.flr_1_16(x3.detach(), pp_inst_feat)   # [B,128,H/16,W/16]
        #     #x4_T = self.flr_1_32(x4.detach(), pp_inst_feat)

        #     def feat_loss(student_f, teacher_f, vmask):
        #         """
        #         student_f : [B,C,H,W]
        #         teacher_f : [B,C,H,W]
        #         vmask     : [B,1,H,W] (0/1)
        #         Smooth L1 + valid mask
        #         """
        #         # [B,1,H,W] -> {0,1}
        #         v = (vmask > 0.5).float()

        #         # [B,C,H,W], v는 브로드캐스트
        #         diff = (student_f - teacher_f) * v

        #         # Smooth L1 (Huber) 부분
        #         abs_diff    = diff.abs()
        #         smooth_mask = (abs_diff < 1.0).float()

        #         value = 0.5 * (diff ** 2) * smooth_mask + \
        #                 (abs_diff - 0.5) * (1.0 - smooth_mask)

        #         # valid element 개수 = (공간 valid 픽셀 수) * 채널 수
        #         denom = v.sum() * student_f.shape[1] + 1e-6

        #         return value.sum() / denom

        #     # L_1_2 = feat_loss(x0, x0_T, valid_1_2)
        #     #L_1_4  = feat_loss(x1, x1_T, valid_1_4)
        #     #L_1_8  = feat_loss(x2, x2_T, valid_1_8)
        #     #L_1_16 = feat_loss(x3, x3_T, valid_1_16)
        #     # L_1_32 = feat_loss(x4, x4_T, valid_1_32)

        #     # 가중치는 일단 0.1씩, 나중에 튜닝
        #     #flr_loss = 1.0 * (L_1_8 + L_1_4)

        lambda_distill = 1
        lambda_llr = 0.7
        lambda_seg = 0.5

        loss = l1_loss + lambda_distill * distill_loss + lambda_llr * llr_loss + lambda_seg * seg_loss


        # ---------------------- 시각화 (train_vis) ----------------------

        vis_dir  = mini_batch_data.get("vis_dir", None)
        vis_name = mini_batch_data.get("vis_name", None)

        if (vis_dir is not None) and (vis_name is not None):
            try:
                os.makedirs(vis_dir, exist_ok=True)

                # 원본 크기 (padding 전)
                _, _, H0, W0 = mini_batch_data["img"].shape

                # 배치에서 첫 번째 샘플만 사용
                img_vis   = img[0].detach().cpu()[:, :H0, :W0]        # [3, H, W] BGR
                label_vis = label[0].detach().cpu()[:, :H0, :W0]      # [1, H, W]
                pred_vis  = pred[0].detach().cpu()[:, :H0, :W0]       # [1, H, W]

                # ---- 1) RGB 이미지: 정규화 없이 raw BGR 0~255 기준 ----
                img_bgr = img_vis.permute(1, 2, 0).numpy()            # [H, W, 3]
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)   # float일 수 있으니 uint8로

                def depth_to_color(d_tensor, max_depth=80.0):
                    d = d_tensor.squeeze(0).numpy()   # [H, W], float

                    if np.all(d == 0):
                        norm = np.zeros_like(d, dtype=np.uint8)
                    else:
                        # 0~max_depth로 클리핑하고 정규화
                        d_clipped = np.clip(d, 0, max_depth) / max_depth
                        # 가까운=빨강, 먼=파랑이 되도록 반전
                        d_inv = 1.0 - d_clipped
                        norm = (d_inv * 255).astype(np.uint8)

                    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    return color

                gt_color     = depth_to_color(label_vis)      # LiDAR GT depth
                pred_color   = depth_to_color(pred_vis)       # Student pred depth

                # 🔹 1) Teacher depth 시각화
                teacher_vis  = teacher[0].detach().cpu()[:, :H0, :W0]     # [1,H,W]
                teacher_color = depth_to_color(teacher_vis)               # [H,W,3]

                # 🔹 2) Panoptic semantic 시각화
                pp_sem_vis   = pp_sem[0].detach().cpu()[:, :H0, :W0]      # [1,H,W]
                pp_vm_vis    = pp_valid_mask[0].detach().cpu()[:, :H0, :W0]  # [1, H, W]

                sem = pp_sem_vis.squeeze(0).numpy().astype(np.int32)
                vm  = pp_vm_vis.squeeze(0).numpy().astype(bool)

                sem_vis = sem.copy()
                sem_vis[sem_vis == 255] = 0    # void 제거

                if sem_vis.max() == 0:
                    sem_color = np.zeros((H0, W0, 3), dtype=np.uint8)
                else:
                    max_valid = sem_vis.max()  # 여기서는 5 정도
                    sem_norm = (sem_vis.astype(np.float32) / max_valid) * 255.0
                    sem_norm = sem_norm.astype(np.uint8)
                    sem_color = cv2.applyColorMap(sem_norm, cv2.COLORMAP_JET)
                    sem_color[~vm] = 0


                # ---- 3) 네 장을 옆으로 이어붙이기 ----
                panel = np.concatenate(
                    [img_bgr, sem_color, teacher_color, pred_color], axis=1
                )

                out_path = os.path.join(vis_dir, vis_name)
                cv2.imwrite(out_path, panel)

            except Exception as e:
                # 시각화 쪽 문제 때문에 학습이 멈추지 않도록 조용히 무시
                pass

        with torch.no_grad():
            mae = (torch.abs(pred - label) * label_mask).sum() / (label_mask.sum() + 1e-6)
            rmse = torch.sqrt(((pred - label) ** 2 * label_mask).sum() / (label_mask.sum() + 1e-6))

        monitors = {
            "train_loss":  loss.detach(),
            "mae":         mae.detach(),
            "rmse":        rmse.detach(),
            "flr_loss":    flr_loss.detach(),
            "distill_loss":distill_loss.detach(),
            "llr_loss":    llr_loss.detach(),
            "seg_loss":    seg_loss.detach(),
        }

        return loss, monitors

        # ----------------------------------------------------------------

    def forward_eval(self, mini_batch_data: dict):
        """
        valid / test DataLoader에서 나온 mini_batch_data 하나를 받아서
        - padding + device/dtype 맞추고
        - 모델 추론
        - 원래 H, W로 크롭해서 dict로 반환
        """
        # 원본 크기 (padding 하기 전 size를 기억해 둠)
        B, C, H, W = mini_batch_data["img"].shape

        # 필수 항목
        img        = self.padding(self.make_torch_tensor(mini_batch_data["img"],   self.device, self.dtype))
        label      = self.padding(self.make_torch_tensor(mini_batch_data["label"], self.device, self.dtype))
        label_mask = self.padding(self.make_torch_tensor(mini_batch_data["label_mask"], self.device, self.dtype))
        #pp_inst = self.padding(self.make_torch_tensor(mini_batch_data['pp_inst'], self.device, torch.long))
        #pp_sem = self.padding(self.make_torch_tensor(mini_batch_data['pp_sem'], self.device, torch.long))
        #pp_valid_mask = self.padding(self.make_torch_tensor(mini_batch_data['pp_valid_mask'], self.device, self.dtype))

 
        with torch.no_grad():
            # 단일 스케일 depth 예측
             pred, _ = self.forward(img)   # [B, 1, H_pad, W_pad]

        out = {
            "pred":       pred[:, :, :H, :W],       # [B, 1, H, W]
            "img":        img[:, :, :H, :W],        # [B, 3, H, W]
            "label":      label[:, :, :H, :W],      # [B, 1, H, W]
            "label_mask": label_mask[:, :, :H, :W], # [B, 1, H, W]
        }

        return out

    def forward_test(self,img):
        """
        img: [B, 3, H, W] (np.ndarray or torch.Tensor)
        return: depth_pred: [B, 1, H, W] (numpy)
        """
        B, C, H, W = img.shape

        img = self.padding(self.make_torch_tensor(img, self.device, self.dtype))

        with torch.no_grad():
             pred, _ = self.forward(img)   # [B, 1, H_pad, W_pad]

        # padding 전에 원래 크기만 잘라서 numpy로 반환
        pred = pred[:, :, :H, :W].detach().cpu().numpy()
        return pred

