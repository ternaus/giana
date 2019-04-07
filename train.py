"""Main train loop."""

import argparse
import json
from pathlib import Path
from dataset_utils.get_splits import get_train_val_image_paths
from catalyst.dl.experiments import SupervisedRunner
import torch
from models import UNet11
from data_loaders import GianaDataset
from torch.utils.data import DataLoader
from collections import OrderedDict  # noqa F401
import augmentations
from torch import nn
from losses import LossBinary

model_names = {'UNet11': UNet11}


def get_loader(file_names, shuffle=False, transform=None, batch_size=1, num_workers=1):
    return DataLoader(
        dataset=GianaDataset(file_names, transform=transform),
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )


def get_model(model_name, device_ids):
    # model, criterion, optimizer
    device_ids = list(map(int, device_ids.split(',')))
    model = model_names[model_name]

    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    return model.cuda()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-c', '--config_path', default='configs/dataset.json', type=str)
    arg('-f', '--fold_id', default=0, type=int, help='Fold id.')
    arg('-l', '--log_dir', default='./logdir', type=str, help='Path to store logs')
    arg('-n', '--num_epochs', default=42, type=int, help='Number of epochs.')
    arg('-m', '--model_name', default='UNet11', type=str, help='Name of the network')
    arg('-j', '--num_workers', default=1, type=int, help='Number of CPU threads to use.')
    arg('-b', '--batch_size', default=1, type=int, help='Size of the batch')
    arg('-d', '--device_ids', default='0', type=str, help='GPU device ids to use.')
    arg('-j', '--jaccard_weight', default=0.3, type=float, help='Weight for soft Jaccard in loss.')
    arg('--num_folds', default=5, type=int, help='Number of folds.')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    data_path = Path(config['data_path']).absolute().expanduser()

    train_file_names, val_files_names = get_train_val_image_paths(data_path / 'train', args.fold_id, args.num_folds)

    train_transform = augmentations.get_train_transform()
    val_transform = augmentations.get_val_transform()

    train_loader = get_loader(train_file_names, train_transform, args.num_workers)
    val_loader = get_loader(val_files_names, val_transform, args.num_workers)

    # data
    loaders = OrderedDict({"train": train_loader, "valid": val_loader})

    # model, criterion, optimizer
    model = get_model(args.model_name, args.device_ids)

    criterion = LossBinary(jaccard_weight=args.jaccard_weight)
    optimizer = torch.optim.Adam(model.parameters())

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=args.logdir,
        num_epochs=args.num_epochs,
        verbose=True
    )


if __name__ == '__main__':
    main()
