#!/usr/bin/env python3
import os
import json

import cv2
import numpy as np
import pickle

from PIL import Image

import torch
from torch.utils.data import Dataset

from nuscenes.utils.data_classes import LidarPointCloud
from mapillary_coarse8 import remap_semantic_to_8, IGNORE_INDEX, NUM_SEM_CLASSES

from utils import (
    project_3d_to_2d,
    get_depth_map,
    colorize_depth_map,  # __main__에서 쓰므로 여기로 올림
)


class conf:
    input_h, input_w = 900, 1600
    max_depth = 80
    min_depth = 0
    query_radius = 3.0


rng = np.random.default_rng()


def random_hflip(img, depth, depth_mask, p=0.5, extra=None):
    """
    img, depth, depth_mask: (C, H, W) numpy 배열
    extra: 같이 뒤집어야 하는 추가 맵 리스트 (각각 (C, H, W))
    """
    if extra is None:
        extra = []

    if rng.uniform(0.0, 1.0) > p:
        return img, depth, depth_mask, extra

    # numpy flip + copy 로 stride 양수로 보정
    img = np.flip(img, axis=2).copy()          # W 축
    depth = np.flip(depth, axis=2).copy()
    depth_mask = np.flip(depth_mask, axis=2).copy()
    extra_flipped = [np.flip(e, axis=2).copy() for e in extra]

    return img, depth, depth_mask, extra_flipped


class Vidar(torch.utils.data.Dataset):

    path = '/mnt/nas/mlmlab/data/nuscenes/nuscenes_infos_train.pkl'

    data_root = '/mnt/nas/mlmlab/data/nuscenes/'

    mono_depth_root = '/mnt/nas/mlmlab/data/mono_depth_pro'

    panoptic_root = '/home/jhkim/jeomyo/pseudo_panoptic'

    def __init__(self):
        with open(self.path, 'rb') as f:
            infos = pickle.load(f)

        # 우리가 확인한 구조: {'metainfo': ..., 'data_list': [...]}
        if isinstance(infos, dict) and 'data_list' in infos:
            self.infos = infos['data_list']
        else:
            self.infos = infos

        # DLR-lite: only front camera & lidar
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'

    def __len__(self):
        return len(self.infos)

    # -----------------------------
    # Panoptic 로딩 helper
    # -----------------------------
    def load_panoptic(self, cam_info, depth_mask_np):
        """
        cam_info와 depth_mask를 받아서
        - pp_inst: (1, H, W) int32   (instance id, 0 = no instance/sky/void)
        - pp_sem:  (1, H, W) int16   (category_id, 0 = unknown/없음)
        - pp_valid_mask: (1, H, W) uint8 (depth 유효 & instance 존재하는 픽셀)
        을 리턴한다.

        전제:
        - sky 제거 스크립트를 이미 돌려서 sky segment는 JSON/PNG에서 빠져있다.
        - panoptic PNG / JSON 해상도는 CAM_FRONT 이미지와 동일하다.
        - 파일 이름 규칙:
          <img_basename>_panoptic.json
          <img_basename>_panopticId.png
        """
        base = os.path.basename(cam_info['img_path'])  # 예: n015-...jpg
        stem, _ = os.path.splitext(base)

        json_path = os.path.join(self.panoptic_root, stem + "_panoptic.json")
        png_path = os.path.join(self.panoptic_root, stem + "_panopticId.png")

        if (not os.path.exists(json_path)) or (not os.path.exists(png_path)):
            # panoptic이 없는 경우: 전부 0으로 채우고 valid_mask도 0
            _, H, W = depth_mask_np.shape
            pp_inst = np.zeros((1, H, W), dtype=np.int32)
            pp_sem = np.zeros((1, H, W), dtype=np.int16)
            pp_valid_mask = np.zeros((1, H, W), dtype=np.uint8)
            return pp_inst, pp_sem, pp_valid_mask

        # JSON 로드
        with open(json_path, "r") as f:
            ann = json.load(f)
        segments = ann.get("segments_info", [])

        # PNG 로드: (H, W)
        pan = np.array(Image.open(png_path)).astype(np.int32)
        H_pan, W_pan = pan.shape[:2]

        # depth_mask와 해상도 맞는지 확인
        _, H_d, W_d = depth_mask_np.shape
        assert (H_pan, W_pan) == (H_d, W_d), \
            f"Panoptic size {H_pan,W_pan} != depth size {H_d,W_d} for {png_path}"

        # instance id map
        inst_map = pan  # (H, W), int32

        # semantic id map
        sem_map = np.zeros_like(inst_map, dtype=np.int16)
        for seg in segments:
            sid = seg["id"]
            cid = seg["category_id"]
            mask = (inst_map == sid)
            sem_map[mask] = cid

        sem_map_8 = remap_semantic_to_8(sem_map, ignore_index=IGNORE_INDEX) 

        # instance가 존재하는 픽셀 (0은 no-instance / sky / void 등)
        has_inst = (inst_map > 0).astype(np.uint8)  # (H, W)

        # depth_mask_np: (1, H, W) → (H, W)
        depth_valid = (depth_mask_np[0] > 0).astype(np.uint8)

        # 최종 valid mask: depth도 있고 instance도 있는 픽셀만 1
        valid_mask = has_inst.astype(np.uint8)  # (H, W)
        valid_mask[sem_map_8 == IGNORE_INDEX] = 0

        # CHW로 reshape
        pp_inst = inst_map.astype(np.int32)[None, ...]          # (1, H, W)
        pp_sem  = sem_map_8.astype(np.int16)[None, ...]            # (1, H, W)
        pp_valid_mask = valid_mask.astype(np.uint8)[None, ...]  # (1, H, W)

        return pp_inst, pp_sem, pp_valid_mask

    def __getitem__(self, index):
        data = self.infos[index]

        # ============================
        # 1) CAM_FRONT 이미지 로드 (mmdet3d-style)
        # ============================
        cam_info = data['images'][self.camera_use_type]

        # img_path: 'n015-...jpg' → <root>/samples/CAM_FRONT/<img_path>
        img_path = os.path.join(
            self.data_root, 'samples', self.camera_use_type, cam_info['img_path']
        )
        img = cv2.imread(img_path)  # BGR, HxWx3
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        H, W = img.shape[:2]

        # ============================
        # 2) LIDAR_TOP 로드
        # ============================
        lidar_info = data['lidar_points']
        lidar_path = os.path.join(
            self.data_root, 'samples', self.lidar_use_type, lidar_info['lidar_path']
        )

        lidar_obj = LidarPointCloud.from_file(lidar_path)
        pts_lidar = lidar_obj.points[:3, :].T  # (N, 3) xyz (LiDAR frame)

        # 동차 좌표 (x, y, z, 1)
        N = pts_lidar.shape[0]
        pts_h = np.concatenate(
            [pts_lidar, np.ones((N, 1), dtype=pts_lidar.dtype)],
            axis=1
        )  # (N, 4)

        # ============================
        # 3) LiDAR → Camera 변환 (lidar2cam + cam2img)
        # ============================
        lidar2cam = np.array(cam_info['lidar2cam'], dtype=np.float32)  # (4, 4)
        cam2img = np.array(cam_info['cam2img'], dtype=np.float32)      # (3, 3)

        pts_cam = (lidar2cam @ pts_h.T).T   # (N, 4)
        xyz_cam = pts_cam[:, :3]
        z_cam = xyz_cam[:, 2]

        # 깊이 범위 필터
        valid = (z_cam > conf.min_depth) & (z_cam < conf.max_depth)
        xyz_cam = xyz_cam[valid]
        z_cam = z_cam[valid]

        # 카메라 평면으로 투영
        if xyz_cam.shape[0] == 0:
            # 이 프레임에 유효 LiDAR 포인트가 0개인 경우 방어
            depth_np = np.zeros((H, W), dtype=np.float32)
        else:
            uv = project_3d_to_2d(xyz_cam, cam2img)  # (N, 2) [u, v]
            # (u, v, depth) → depth map 생성
            lidar_uvd = np.concatenate([uv, z_cam[:, None]], axis=1)  # (N, 3)
            depth_np = get_depth_map(lidar_uvd, (H, W))  # (H, W), float32

        # ============================
        # 4) numpy / shape 정리
        # ============================
        # img: BGR(H,W,3) → CHW (3,H,W)
        img_np = np.ascontiguousarray(img.transpose(2, 0, 1))  # (3, H, W)

        depth_mask_np = (depth_np > 0).astype(np.uint8)
        depth_np = depth_np.astype('float32')

        depth_np = depth_np[None, ...]           # (1, H, W)
        depth_mask_np = depth_mask_np[None, ...] # (1, H, W)

        # ============================
        # 4.5) DepthPro teacher depth 로드
        # ============================
        # img 파일 이름에서 stem 추출
        img_base = os.path.basename(cam_info['img_path'])  # n008-...__CAM_FRONT__1526915243012465.jpg
        stem, _ = os.path.splitext(img_base)               # n008-...__CAM_FRONT__1526915243012465

        teacher_path = os.path.join(self.mono_depth_root, stem + '.npy')

        if os.path.exists(teacher_path):
            teacher_np = np.load(teacher_path).astype('float32')  # (H, W) or (1,H,W)
            if teacher_np.ndim == 2:
                teacher_np = teacher_np[None, ...]  # (1, H, W)
        else:
            # 없으면 0으로 채우기
            _, H, W = depth_np.shape
            teacher_np = np.zeros((1, H, W), dtype=np.float32)

        teacher_np = np.clip(teacher_np, conf.min_depth, conf.max_depth)

        # ============================
        # 5) Panoptic pseudo-GT 로드
        # ============================
        pp_inst_np, pp_sem_np, pp_valid_mask_np = self.load_panoptic(
            cam_info, depth_mask_np
        )

        # ============================
        # 6) Augmentation: Horizontal Flip
        # ============================
        extras = [teacher_np, pp_inst_np, pp_sem_np, pp_valid_mask_np]
        img_np, depth_np, depth_mask_np, extras = random_hflip(
            img_np, depth_np, depth_mask_np, p=0.5, extra=extras
        )
        teacher_np, pp_inst_np, pp_sem_np, pp_valid_mask_np = extras

        # ============================
        # 7) 튜플로 반환
        # ============================
        # 튜플 순서:
        #   0: img           (3, H, W)
        #   1: depth         (1, H, W)
        #   2: depth_mask    (1, H, W)
        #   3: pp_inst       (1, H, W)
        #   4: pp_sem        (1, H, W)
        #   5: pp_valid_mask (1, H, W)
        return (
            img_np,
            depth_np,
            depth_mask_np,
            pp_inst_np,
            pp_sem_np,
            pp_valid_mask_np,
            teacher_np,         # 🔹 추가: (1, H, W)
        )


if __name__ == '__main__':

    def make_depth_overlay(img_chw, depth_1hw, mask_1hw, alpha=0.6):
        """
        img_chw: (3, H, W) BGR
        depth_1hw: (1, H, W) float32 depth (meters)
        mask_1hw: (1, H, W) uint8 (0/1)
        """
        img_vis = img_chw.transpose(1, 2, 0).astype(np.float32)  # (H, W, 3)
        d = depth_1hw.squeeze()                                  # (H, W)
        m = mask_1hw.squeeze().astype(bool)                      # (H, W)

        d_clipped = np.clip(d, 0, 80) / 80.0
        depth_color = colorize_depth_map(d_clipped).astype(np.float32)  # (H,W,3)

        overlay = img_vis.copy()
        overlay[m] = (1 - alpha) * img_vis[m] + alpha * depth_color[m]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return img_vis.astype(np.uint8), depth_color.astype(np.uint8), overlay

    def visualize_panoptic_sem(pp_sem_1hw, valid_mask_1hw=None):
        sem = pp_sem_1hw.squeeze().astype(np.int32)  # (H, W)

        # 1) 시각화용 복사본 만들고 IGNORE는 0으로 처리
        sem_vis = sem.copy()
        sem_vis[sem_vis == IGNORE_INDEX] = 0

        # 2) 0 ~ NUM_SEM_CLASSES-1 범위 기준으로 정규화
        max_label = max(1, NUM_SEM_CLASSES - 1)  # 예: 8-class면 7
        sem_norm = (sem_vis.clip(0, max_label).astype(np.float32) / max_label) * 255.0
        sem_norm = sem_norm.astype(np.uint8)

        sem_color = cv2.applyColorMap(sem_norm, cv2.COLORMAP_JET)

        # 3) valid_mask 있으면 그 외는 까맣게
        if valid_mask_1hw is not None:
            vm = valid_mask_1hw.squeeze().astype(bool)
            sem_color[~vm] = 0

        return sem_color

    def overlay_mask_on_img(img_hw3, mask_hw, color=(0, 0, 255), alpha=0.5):
        """
        img_hw3: (H, W, 3) BGR
        mask_hw: (H, W) uint8 0/1
        """
        img = img_hw3.astype(np.float32).copy()
        mask = mask_hw.astype(bool)

        overlay = img.copy()
        color_arr = np.zeros_like(img)
        color_arr[:, :] = np.array(color, dtype=np.float32)

        overlay[mask] = (1 - alpha) * img[mask] + alpha * color_arr[mask]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    # ============================
    # 0) 샘플 하나 뽑기
    # ============================
    dataset = Vidar()
    (
        img,           # (3, H, W), BGR
        depth,         # (1, H, W)
        depth_mask,    # (1, H, W)
        pp_inst,       # (1, H, W)
        pp_sem,        # (1, H, W)
        pp_valid_mask, # (1, H, W)
        teacher_depth, # (1, H, W)  🔹 추가
    ) = dataset[0]


    # ============================
    # 0-1) 기본 통계 출력 (shape / dtype / 값 분포)
    # ============================
    print("img shape, dtype:", img.shape, img.dtype)
    print("depth shape, dtype:", depth.shape, depth.dtype)
    print("depth_mask shape, dtype:", depth_mask.shape, depth_mask.dtype)
    print("pp_inst shape, dtype:", pp_inst.shape, pp_inst.dtype)
    print("pp_sem shape, dtype:", pp_sem.shape, pp_sem.dtype)
    print("pp_valid_mask shape, dtype:", pp_valid_mask.shape, pp_valid_mask.dtype)
    print("teacher_depth shape, dtype:", teacher_depth.shape, teacher_depth.dtype)

    d = depth.squeeze()
    dm = depth_mask.squeeze()
    print("depth stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        float(d.min()), float(d.max()), float(d.mean())
    ))
    td = teacher_depth.squeeze()
    print("teacher_depth stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        float(td.min()), float(td.max()), float(td.mean())
    ))
    print("depth non-zero count:", int((d > 0).sum()))
    print("depth_mask unique:", np.unique(dm))

    print("pp_inst unique:", np.unique(pp_inst))
    print("pp_sem unique:", np.unique(pp_sem))
    print("pp_valid_mask unique:", np.unique(pp_valid_mask))

    # ============================
    # 1) depth overlay / raw 저장
    # ============================
    img_vis, depth_color, overlay = make_depth_overlay(img, depth, depth_mask)

    cv2.imwrite('raw_img.png', img_vis)                # BGR
    cv2.imwrite('raw_depth_color.png', depth_color)    # depth colormap
    cv2.imwrite('overlay_depth_on_img.png', overlay)   # overlay

    # ============================
    # 1-1) DepthPro teacher depth overlay / raw 저장
    # ============================
    # teacher_depth는 dense일 수 있어서 mask는 전체 1로 두거나, (teacher_depth > 0)로 줘도 됨
    teacher_mask = (teacher_depth > 0).astype(np.uint8)

    img_vis_t, depth_color_t, overlay_t = make_depth_overlay(
        img, teacher_depth, teacher_mask
    )

    cv2.imwrite('teacher_depth_color.png', depth_color_t)
    cv2.imwrite('overlay_teacher_depth_on_img.png', overlay_t)

    # ============================
    # 2) panoptic semantic 시각화
    # ============================
    sem_color_full = visualize_panoptic_sem(pp_sem)  # valid_mask 없이 전체 카테고리
    cv2.imwrite('panoptic_sem_color_full.png', sem_color_full)

    # depth ∩ instance valid 영역만 표시
    sem_color_valid = visualize_panoptic_sem(pp_sem, pp_valid_mask)
    cv2.imwrite('panoptic_sem_color_valid.png', sem_color_valid)

    # ============================
    # 3) pp_valid_mask overlay 시각화
    # ============================
    vm = pp_valid_mask.squeeze().astype(np.uint8)  # (H, W)
    vm_overlay = overlay_mask_on_img(img_vis, vm, color=(0, 0, 255), alpha=0.5)
    cv2.imwrite('panoptic_valid_overlay.png', vm_overlay)

    # ============================
    # 4) flip 함수 검증용: no-flip vs 강제 flip
    # ============================
    base_img    = img.copy()
    base_depth  = depth.copy()
    base_mask   = depth_mask.copy()
    base_extras = [pp_inst.copy(), pp_sem.copy(), pp_valid_mask.copy()]

    # (a) p=0.0 → 절대 뒤집지 않음
    img_nf, depth_nf, mask_nf, extras_nf = random_hflip(
        base_img.copy(),
        base_depth.copy(),
        base_mask.copy(),
        p=0.0,
        extra=[e.copy() for e in base_extras],
    )
    pp_inst_nf, pp_sem_nf, pp_valid_nf = extras_nf

    img_nf_vis, depth_nf_color, overlay_nf = make_depth_overlay(
        img_nf, depth_nf, mask_nf
    )
    cv2.imwrite('noflip_overlay.png', overlay_nf)

    # (b) p=1.0 → 항상 강제 flip
    img_ff, depth_ff, mask_ff, extras_ff = random_hflip(
        base_img.copy(),
        base_depth.copy(),
        base_mask.copy(),
        p=1.0,
        extra=[e.copy() for e in base_extras],
    )
    pp_inst_ff, pp_sem_ff, pp_valid_ff = extras_ff

    img_ff_vis, depth_ff_color, overlay_ff = make_depth_overlay(
        img_ff, depth_ff, mask_ff
    )
    cv2.imwrite('flip_overlay.png', overlay_ff)

    # panoptic valid mask도 flip된 버전으로 overlay 비교
    vm_nf = pp_valid_nf.squeeze().astype(np.uint8)
    vm_ff = pp_valid_ff.squeeze().astype(np.uint8)

    vm_nf_overlay = overlay_mask_on_img(
        img_nf_vis, vm_nf, color=(0, 0, 255), alpha=0.5
    )
    vm_ff_overlay = overlay_mask_on_img(
        img_ff_vis, vm_ff, color=(0, 255, 0), alpha=0.5
    )

    cv2.imwrite('noflip_panoptic_valid_overlay.png', vm_nf_overlay)
    cv2.imwrite('flip_panoptic_valid_overlay.png', vm_ff_overlay)

    print("Saved:")
    print("  raw_img.png")
    print("  raw_depth_color.png")
    print("  overlay_depth_on_img.png")
    print("  panoptic_sem_color_full.png")
    print("  panoptic_sem_color_valid.png")
    print("  panoptic_valid_overlay.png")
    print("  noflip_overlay.png / flip_overlay.png")
    print("  noflip_panoptic_valid_overlay.png / flip_panoptic_valid_overlay.png")
