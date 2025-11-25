#!/usr/bin/env python3
import argparse
from io import BytesIO
import os
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
from pathlib import Path
import sys
import time
import traceback
from loguru import logger
logger.remove()
if(LOCAL_RANK == 0):
    logger.add(sys.stdout, colorize=True, level="INFO", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
else:
    logger.add(sys.stderr, colorize=True, level="ERROR", 
        format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import Network
import dataset
from utils import (
    TrainClock,
    log_rate_limited,
)


def ensure_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


class config(dataset.conf):
    base_lr = 3e-3
    epoch_num = 15
    checkpoint_interval = 1
    log_interval = 20

    exp_dir = os.path.dirname(__file__)
    exp_name = os.path.basename(exp_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  
    local_train_log_path = f'./train_log/run_{timestamp}'
    log_dir = str(local_train_log_path)
    log_model_dir = os.path.join(local_train_log_path, 'models')
    
    params = {
        'batch_size': 10,
        'shuffle': False,    # sampler가 셔플 담당
        'num_workers': 4,
        'persistent_workers': True
    }


class Session:
    def __init__(self, config, net=None, rank=0, local_rank=0):
        self.log_dir = config.log_dir
        ensure_dir(self.log_dir)
        self.model_dir = config.log_model_dir
        ensure_dir(self.model_dir)

        self.clock = TrainClock()
        self.config = config
        self.lr_scheduler = None
        self.net = net
        self.optimizer = None
        self.rank = rank
        self.local_rank = local_rank

    def start(self):
        self.save_checkpoint('start')

    def save_checkpoint(self, name):
        if self.rank != 0:
            return
        net = self.net.module if isinstance(self.net, DDP) else self.net
        net_state = net.state_dict()

        ckp = {
            'network': net_state,
            'clock': self.clock.make_checkpoint(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(ckp, Path(self.config.log_model_dir) / (name + '.ckpt'))

        torch.save({"network": net_state},
                   Path(self.config.log_model_dir) / (name + '.net.ckpt'))

    def load_misc_checkpoint(self, ckp_path:Path):
        checkpoint = torch.load(
            ckp_path, 
            map_location=torch.device(f"cuda:{self.local_rank}")
        )
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def load_net_state_dict(self, ckp_path:Path):
        if self.rank == 0:
            checkpoint = torch.load(
                ckp_path, 
                map_location=torch.device(f"cuda:{self.local_rank}")
            )
            self.net.load_state_dict(checkpoint['network'], strict=False)


def main():
    parser = argparse.ArgumentParser()
    default_devices = '*' if os.environ.get('RLAUNCH_WORKER') else '0'
    parser.add_argument('-d', '--device', default=default_devices)
    parser.add_argument('--fast-run', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('-r', '--restart', action='store_true')
    args = parser.parse_args()

    if(LOCAL_RANK == 0):
        log_path = Path(config.log_dir) / "worklog.log"
        logger.add(str(log_path.resolve()), colorize=True, level="INFO", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")
    else:
        log_path = Path(config.log_dir) / f"worklog_{RANK}.log"
        logger.add(str(log_path.resolve()), colorize=True, level="ERROR", 
            format="<green>[{time:%m-%d %H:%M:%S}]</green> {message}")

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl')

    net = Network().cuda(LOCAL_RANK) 
    sess = Session(config, net=net, rank=RANK, local_rank=LOCAL_RANK)
    clock = sess.clock

    continue_path = Path(config.log_model_dir) / "latest" if args.restart else None

    # load network weights (only rank 0)
    if continue_path and continue_path.with_name("latest.net.ckpt").exists() and RANK == 0:
        sess.load_net_state_dict(continue_path.with_name("latest.net.ckpt"))

    torch.distributed.barrier()

    # DDP wrapping
    if torch.cuda.device_count() > 1:
        logger.info("Using DDP!")
        net = DDP(sess.net,
                  device_ids=[LOCAL_RANK],
                  output_device=LOCAL_RANK,
                  find_unused_parameters=True)
        sess.net = net

    # ---------------------------
    # Dataset + Sampler
    # ---------------------------
    dataset_train = dataset.Vidar()

    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True
    )

    train_ds = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.params['batch_size'],
        num_workers=config.params['num_workers'],
        persistent_workers=config.params['persistent_workers'],
        sampler=train_sampler,
        shuffle=False
    )

    # ---------------------------
    # Optimizer + LR scheduler
    # ---------------------------
    opt = torch.optim.AdamW(sess.net.parameters(), lr=1e-3, weight_decay=4e-8)

    total_step = len(train_ds) * config.epoch_num
    base_lr = config.base_lr

    # def lr_func(step):
    #     return base_lr * (np.cos(step / total_step * np.pi) + 1) * 0.5 + 1e-3

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_func)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)

    sess.optimizer = opt
    sess.lr_scheduler = lr_scheduler

    if RANK == 0:
        core_model = sess.net.module if hasattr(sess.net, "module") else sess.net
        num_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)

        logger.info("========== Training Hyper-Params ==========")
        logger.info(f"exp_name     : {config.exp_name}")
        logger.info(f"log_dir      : {config.log_dir}")
        logger.info(f"world_size   : {WORLD_SIZE}")
        logger.info(f"batch_size   : {config.params['batch_size']}")
        logger.info(f"epochs       : {config.epoch_num}")
        logger.info(f"base_lr      : {config.base_lr}")
        logger.info(f"weight_decay : {4e-8}")
        logger.info(f"pad_divisor  : {getattr(core_model, 'pad_divisor', 'N/A')}")
        logger.info(f"#params      : {num_params/1e6:.2f} M")
        logger.info("===========================================")

    # Restore training state except network
    if continue_path and continue_path.with_name("latest.ckpt").exists():
        sess.load_misc_checkpoint(continue_path.with_name("latest.ckpt"))

    sess.start()
    log_output = log_rate_limited(min_interval=1)(logger.info)

    step_start = clock.step
    loss_record, monitors_record = 0, {}

    time_train_start = time.time()

    # train 중 시각화 설정
    VIS_INTERVAL = 50  # 몇 step마다 한 번씩 저장할지 (원하면 100, 1000 등으로 조절)
    VIS_DIR = os.path.join(config.log_dir, "train_vis")

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(config.epoch_num):

        train_sampler.set_epoch(epoch)

        net.train()
        time_iter_start = time.time()

        for idx, batch in enumerate(train_ds):
            tdata = time.time() - time_iter_start

            img, depth, depth_mask, pp_inst, pp_sem, pp_valid_mask, teacher_depth = batch

            mini_batch_data = {
                'img': img,
                'label': depth,
                'label_mask': depth_mask,
                'pp_inst': pp_inst,
                'pp_sem': pp_sem,
                'pp_valid_mask': pp_valid_mask,
                'teacher_depth': teacher_depth, 
            }


            # ---- train 중 시각화: rank 0에서만, VIS_INTERVAL마다 한 번 ----
            if (RANK == 0) and (idx % VIS_INTERVAL == 0):
                mini_batch_data["vis_dir"] = VIS_DIR
                mini_batch_data["vis_name"] = f"e{clock.epoch:02d}_i{idx:06d}.png"
            else:
                mini_batch_data["vis_dir"] = None
                mini_batch_data["vis_name"] = None

            # ★ DDP 여부와 무관하게 forward_train 호출 가능하도록 수정
            model = sess.net.module if hasattr(sess.net, 'module') else sess.net  # ★
            loss, monitors = model.forward_train(mini_batch_data)                 # ★

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            # ----------- Logging ----------- #
            loss_record += loss.item()
            time_train_passed = time.time() - time_train_start
            step_passed = clock.step - step_start
            eta = (total_step - clock.epoch) * (time_train_passed / max(step_passed, 1e-7))
            time_iter_passed = time.time() - time_iter_start

            if RANK == 0:
                if monitors:
                    for k, v in monitors.items():
                        monitors_record[k] = monitors_record.setdefault(k, 0) + v

                if idx and (idx + 1) % config.log_interval == 0:
                    loss_record /= config.log_interval
                    if monitors_record:
                        for k in monitors_record:
                            monitors_record[k] /= config.log_interval

                    meta = []
                    meta.append('{:.2g} b/s'.format(1. / time_iter_passed))
                    meta.append('passed:{}'.format(format_time(time_train_passed)))
                    meta.append('eta:{}'.format(format_time(eta)))
                    meta.append('data_time:{:.2%}'.format(tdata / time_iter_passed))
                    meta.append('lr:{:.5g}'.format(lr_scheduler.get_last_lr()[0]))
                    meta.append('[{}:{}/{}]'.format(clock.epoch, idx + 1, len(train_ds)))
                    meta.append('loss:{:.4g}'.format(loss_record))
                    for k, v in monitors_record.items():
                        meta.append(f'{k}:{v:.4g}')

                    log_output(", ".join(meta))
                    loss_record, monitors_record = 0, {}

            time_iter_start = time.time()
            clock.tick()

        clock.tock()

        if RANK == 0:
            if (clock.epoch + 1) % config.checkpoint_interval == 0:
                sess.save_checkpoint(f'epoch-{clock.epoch}')
            sess.save_checkpoint('latest')

    logger.info("Training done.")
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exit.")
        os._exit(0)
