import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '7,8,10,11,12,13,14,15'
import argparse
from multiprocessing import cpu_count
from pathlib import Path
import utils
from dataset import *
from trainer import TrainerVideoText
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=str, help='Experiment to run')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument(
        '--workers', type=int, default=None,
        help='set number of workers (default #CPUs - 1)')
    parser.add_argument(
        '--log_dir', type=str, default="runs/eval",
        help='directory to save/load the runs and logs')
    parser.add_argument(
        "--dataroot", type=str, default="../data",
        help="change datasets root path")
    parser.add_argument(
        "--cuda", action="store_true", help="train on GPUs")
    parser.add_argument(
        "--single_gpu", action="store_true", help="Disable multi-GPU")
    args = parser.parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(0)
    num_workers = min(10, cpu_count() - 1) if args.workers is None else args.workers
    print(f"{num_workers} parallel dataloader workers")
    dataset_path = Path(args.dataroot)
    val_set = ProposalDatasetFeatures(
        cfg.dataset.name, dataset_path, cfg.dataset.features,
        cfg.dataset.train_split, cfg.dataset.max_frames, False, False,
        False, 0)
    val_loader = data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=val_set.collate_fn,
        pin_memory=True)
    trainer = TrainerVideoText(
        args.log_dir, cfg, args.cuda, args.cuda and not args.single_gpu,
        args.checkpoint, False)
    trainer.make_ws_data(val_loader)
    trainer.close()


if __name__ == '__main__':
    main()
