#!/usr/bin/env python3
"""
Evaluation script for depth estimation model.
Usage: python eval.py -m <model_checkpoint.pth> [-d nuscenes]
"""
import os
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud

from model import Network
from utils import project_3d_to_2d, get_depth_map


class conf:
    datasets = {
        'nuscenes': '/mnt/nas/mlmlab/data/nuscenes/nuscenes_infos_val.pkl',
    }
    
    data_root = '/mnt/nas/mlmlab/data/nuscenes/'
    default_dataset = 'nuscenes'
    max_depth = 80
    min_depth = 0


class VidarEval:
    def __init__(self, pkl_path, data_root):
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)
        
        # mmdet3d style pkl 구조 처리
        if isinstance(infos, dict) and 'data_list' in infos:
            self.infos = infos['data_list']
        else:
            self.infos = infos
            
        self.data_root = data_root
        
        # Dataset과 동일하게 설정
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = None

        
    def set_model(self, model_path):
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device)

        self.net = Network().to(self.device)
        self.net.load_state_dict(checkpoint['network'])
        self.net.eval()

        print(f"✓ Model loaded from {model_path}")
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        """Dataset과 동일한 방식으로 데이터 로드"""
        data = self.infos[index]
        
        # ============================
        # 1) CAM_FRONT 이미지
        # ============================
        cam_info = data['images'][self.camera_use_type]
        img_path = os.path.join(
            self.data_root, 'samples', self.camera_use_type, cam_info['img_path']
        )
        
        img = cv2.imread(img_path)  # BGR, (H, W, 3)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        H, W = img.shape[:2]
        
        # ============================
        # 2) LIDAR_TOP (GT depth)
        # ============================
        lidar_info = data['lidar_points']
        lidar_path = os.path.join(
            self.data_root, 'samples', self.lidar_use_type, lidar_info['lidar_path']
        )
        
        lidar_obj = LidarPointCloud.from_file(lidar_path)
        pts_lidar = lidar_obj.points[:3, :].T  # (N, 3)
        
        # 동차 좌표
        N = pts_lidar.shape[0]
        pts_h = np.concatenate([pts_lidar, np.ones((N, 1))], axis=1)  # (N, 4)
        
        # ============================
        # 3) LiDAR → Camera 변환
        # ============================
        lidar2cam = np.array(cam_info['lidar2cam'], dtype=np.float32)
        cam2img = np.array(cam_info['cam2img'], dtype=np.float32)
        
        pts_cam = (lidar2cam @ pts_h.T).T  # (N, 4)
        xyz_cam = pts_cam[:, :3]
        z_cam = xyz_cam[:, 2]
        
        # 깊이 필터링
        valid = (z_cam > conf.min_depth) & (z_cam < conf.max_depth)
        xyz_cam = xyz_cam[valid]
        z_cam = z_cam[valid]
        
        # 투영
        if xyz_cam.shape[0] == 0:
            depth_gt = np.zeros((H, W), dtype=np.float32)
        else:
            uv = project_3d_to_2d(xyz_cam, cam2img)
            lidar_uvd = np.concatenate([uv, z_cam[:, None]], axis=1)
            depth_gt = get_depth_map(lidar_uvd, (H, W))
        
        # ============================
        # 4) 전처리 (Dataset과 동일하게)
        # ============================
        img_np = np.ascontiguousarray(img.transpose(2, 0, 1))  # (3, H, W)
        
        depth_mask_np = (depth_gt > 0).astype(np.uint8)[None, ...]  # (1, H, W)

        return img_np, depth_gt, img_path
    
    def get_metrics(self, pred, gt, mask):
        """Depth estimation 메트릭 계산"""
        diff = pred - gt
        
        mae = np.mean(np.abs(diff[mask]))
        rmse = np.sqrt(np.mean(diff[mask]**2))
        
        # Additional metrics (optional)
        # abs_rel = np.mean(np.abs(diff[mask]) / (gt[mask] + 1e-6))
        # sq_rel = np.mean((diff[mask]**2) / (gt[mask] + 1e-6))
        
        return {
            'mae': mae,
            'rmse': rmse,
        }
    
    def eval(self, model_path=None, max_samples=None, save_vis=False, vis_dir='eval_vis'):
        """Evaluation 수행"""
        if model_path is not None:
            self.set_model(model_path)
        
        if self.net is None:
            raise ValueError("Model not loaded. Call set_model() first.")
        
        # 메트릭 저장
        metrics_all = []      # 0-80m
        metrics_50 = []       # 0-50m
        metrics_70 = []       # 0-70m
        
        # Visualization 디렉토리 생성
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
        
        n_samples = len(self) if max_samples is None else min(max_samples, len(self))
        
        for idx in tqdm(range(n_samples), desc='Evaluating'):
            img_np, depth_gt, img_path = self[idx]

            # ============================
            # label / label_mask 생성
            # ============================
            # depth_gt: (H, W) float32
            label = depth_gt.astype(np.float32)[None, None, ...]          # (1, 1, H, W)
            label_mask = ((depth_gt > 0) & (depth_gt < conf.max_depth)).astype(np.float32)[None, None, ...]

            # ============================
            # Forward pass
            # ============================
            mini_batch = {
                'img': img_np[None, ...],   # (1, 3, H, W)
                'label': label,             # (1, 1, H, W)
                'label_mask': label_mask,   # (1, 1, H, W)
            }
            
            with torch.no_grad():
                out = self.net.forward_eval(mini_batch)  # out: dict 또는 tensor

            # forward_eval이 dict를 리턴하니까 pred만 꺼내줌
            if isinstance(out, dict):
                pred = out["pred"]   # [B, 1, H, W]
            else:
                pred = out           # 혹시 나중에 텐서만 리턴하게 바꿔도 안전

            # Tensor → Numpy
            pred_np = pred[0, 0].detach().cpu().numpy()  # (H, W)
            
            # ============================
            # Metrics 계산
            # ============================
            # Flatten
            pred_flat = pred_np.reshape(-1)
            gt_flat = depth_gt.reshape(-1)
            
            # Masks
            mask_80 = (gt_flat > 0) & (gt_flat <= 80)
            mask_50 = (gt_flat > 0) & (gt_flat <= 50)
            mask_70 = (gt_flat > 0) & (gt_flat <= 70)
            
            if mask_80.sum() > 0:
                metrics_all.append(self.get_metrics(pred_flat, gt_flat, mask_80))
            if mask_50.sum() > 0:
                metrics_50.append(self.get_metrics(pred_flat, gt_flat, mask_50))
            if mask_70.sum() > 0:
                metrics_70.append(self.get_metrics(pred_flat, gt_flat, mask_70))
            
            # ============================
            # Visualization (optional)
            # ============================
            if save_vis and (idx % 10 == 0 or idx < 5):  # 처음 5개 + 10개마다
                basename = os.path.basename(img_path).replace('.jpg', '')
                self._save_vis(img_np, pred_np, depth_gt, 
                              os.path.join(vis_dir, f'{idx:04d}_{basename}.png'))
                
                if idx == 0:
                    print(f"\n✓ Visualization saved to: {vis_dir}/")
                    print(f"  - Combined view: {idx:04d}_{basename}.png")
                    print(f"  - Individual maps: {idx:04d}_{basename}_*.png")
        
        # ============================
        # 평균 메트릭 계산
        # ============================
        def avg_metrics(metric_list):
            if len(metric_list) == 0:
                return {'mae': 0.0, 'rmse': 0.0}
            return {
                k: float(np.mean([m[k] for m in metric_list]))
                for k in metric_list[0].keys()
            }
        
        result = {
            'mae_0-80':  avg_metrics(metrics_all)['mae'],
            'rmse_0-80': avg_metrics(metrics_all)['rmse'],
            'mae_0-50':  avg_metrics(metrics_50)['mae'],
            'rmse_0-50': avg_metrics(metrics_50)['rmse'],
            'mae_0-70':  avg_metrics(metrics_70)['mae'],
            'rmse_0-70': avg_metrics(metrics_70)['rmse'],
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for k, v in result.items():
            print(f"  {k:15s}: {v:.4f}")
        print("="*60)
        
        return result
    
    def _save_vis(self, img_np, pred, gt, save_path):
        """Visualization 저장 - RGB, Pred, GT, Error map"""
        # img: (3, H, W) BGR 0~255 → (H, W, 3)
        img_vis = img_np.transpose(1, 2, 0).astype(np.uint8)
        H, W = img_vis.shape[:2]
        
        # depth colormap 함수
        def depth_to_color(depth, max_val=80.0):
            """Depth를 컬러맵으로 변환 (빨강=가까움, 파랑=멀리)"""
            depth_norm = np.clip(depth / max_val * 255, 0, 255).astype(np.uint8)
            return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        # 1) Prediction depth
        pred_color = depth_to_color(pred)
        
        # 2) Ground truth depth
        gt_color = depth_to_color(gt)
        
        # 3) Error map (absolute difference)
        gt_mask = gt > 0
        error = np.zeros_like(pred)
        error[gt_mask] = np.abs(pred[gt_mask] - gt[gt_mask])
        
        # Error: 0~10m 범위로 스케일링 (10m 이상 에러는 빨간색)
        error_norm = np.clip(error / 10.0 * 255, 0, 255).astype(np.uint8)
        error_color = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
        error_color[~gt_mask] = 0  # GT 없는 곳은 검은색
        
        # 4) Overlay: RGB + Pred (반투명)
        overlay = cv2.addWeighted(img_vis, 0.6, pred_color, 0.4, 0)
        
        # 텍스트 추가
        def add_text(img, text, pos=(10, 30)):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return img
        
        img_vis = add_text(img_vis.copy(), "RGB Image")
        pred_color = add_text(pred_color.copy(), "Prediction")
        gt_color = add_text(gt_color.copy(), "Ground Truth")
        error_color = add_text(error_color.copy(), "Abs Error (m)")
        overlay = add_text(overlay.copy(), "RGB + Pred")
        
        # 통계 정보 추가
        if gt_mask.sum() > 0:
            mae = np.mean(np.abs(error[gt_mask]))
            pred_mean = np.mean(pred[gt_mask])
            gt_mean = np.mean(gt[gt_mask])
            
            stats = [
                f"MAE: {mae:.2f}m",
                f"Pred: {pred_mean:.1f}m",
                f"GT: {gt_mean:.1f}m"
            ]
            for i, text in enumerate(stats):
                add_text(pred_color, text, pos=(10, 70 + i*30))
        
        # 5x1 레이아웃: [RGB | Pred | GT | Error | Overlay]
        row1 = np.hstack([img_vis, pred_color, gt_color])
        row2 = np.hstack([error_color, overlay, np.zeros_like(img_vis)])  # 빈 공간 채우기
        
        # 2x3 레이아웃으로 변경
        vis = np.vstack([row1, row2])
        
        # 저장
        cv2.imwrite(save_path, vis)
        
        # 개별 저장도 원한다면
        base = save_path.replace('.png', '')
        cv2.imwrite(f"{base}_rgb.png", img_vis)
        cv2.imwrite(f"{base}_pred.png", pred_color)
        cv2.imwrite(f"{base}_gt.png", gt_color)
        cv2.imwrite(f"{base}_error.png", error_color)
        cv2.imwrite(f"{base}_overlay.png", overlay)


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth estimation model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('-d', '--dataset', type=str, default=conf.default_dataset,
                       choices=['nuscenes'],
                       help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max number of samples to evaluate (for debugging)')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--vis_dir', type=str, default='eval_vis',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # ============================
    # ⚠️ 경로 확인
    # ============================
    pkl_path = conf.datasets[args.dataset]
    data_root = conf.data_root
    
    if not os.path.exists(pkl_path):
        print(f"❌ ERROR: pkl file not found: {pkl_path}")
        print(f"   Please update conf.datasets in the script.")
        return
    
    if not os.path.exists(data_root):
        print(f"❌ ERROR: data_root not found: {data_root}")
        print(f"   Please update conf.data_root in the script.")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ ERROR: model checkpoint not found: {args.model}")
        return
    
    # ============================
    # Evaluation 실행
    # ============================
    evaluator = VidarEval(pkl_path, data_root)
    evaluator.eval(
        model_path=args.model,
        max_samples=args.max_samples,
        save_vis=args.save_vis,
        vis_dir=args.vis_dir,
    )


if __name__ == '__main__':
    main()
