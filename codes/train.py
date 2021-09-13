# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : train.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/7 下午5:17
@Desc   : Train PointNet, VFE and VFE_LW models on KittiCls train set and validate on KittiCls val set
"""
import argparse
import time
import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import KittiClsDataset
from models import PointNet, VFE, VFE_LW


SEED = 13

date = datetime.date.today()


class Logger(object):
    def __init__(self, filename='logs.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PointNet", help="'PointNet' or 'VFE' or 'VFE_LW'")
    parser.add_argument("--batch_size", type=int, default=108)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decay_lr_factor", type=float, default=0.95)
    parser.add_argument("--decay_lr_every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="../runs/train")
    parser.add_argument("--exp", type=str, default="exp1")

    return parser.parse_args()


def get_weighted_random_sampler(dataset):
    dataset_dir = dataset.root_dir
    vehicle_counts = len(os.listdir(os.path.join(dataset_dir, "Vehicle")))
    pedestrian_counts = len(os.listdir(os.path.join(dataset_dir, "Pedestrian")))
    cyclist_counts = len(os.listdir(os.path.join(dataset_dir, "Cyclist")))
    go_counts = len(os.listdir(os.path.join(dataset_dir, "GO")))

    # Apply WeightedRandomSampler to imbalanced datasets
    class_weights = [
        1 / vehicle_counts,
        1 / pedestrian_counts,
        1 / cyclist_counts,
        1 / go_counts
    ]

    sample_weights = np.zeros(len(dataset))
    sample_weights[:vehicle_counts] = class_weights[0]
    sample_weights[vehicle_counts:vehicle_counts + pedestrian_counts] = class_weights[1]
    sample_weights[vehicle_counts + pedestrian_counts:vehicle_counts + pedestrian_counts + cyclist_counts] = class_weights[2]
    sample_weights[vehicle_counts + pedestrian_counts + cyclist_counts:] = class_weights[3]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    return sampler


def train(train_loader, model, criterion, optimizer, epoch, device, writer, args):
    t0 = time.time()
    # Training
    print("#" * 64)
    print("Training model on train_set")
    model.train()
    print(f"Epoch: [{epoch} / {args.epochs}]...")
    running_train_loss = 0
    running_train_corrects = 0
    tt = time.time()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Set grad to zero
        optimizer.zero_grad()

        # Forward
        out = model(x)
        loss = criterion(out, y)
        _, pred = torch.max(out, 1)
        running_train_loss += loss.item()
        running_train_corrects += torch.sum(pred == y).item()

        # Loss backward
        loss.backward()

        # Update network's param
        optimizer.step()

        # Show logs
        if (i + 1) % 10 == 0:
            ttt = time.time()
            print(
                f"Epoch: [{epoch} / {args.epochs}]\tStep: [{i + 1} / {len(train_loader)}]\tLoss: {loss.item():.4f}\tTime: {(ttt - tt):.2f}s")
            tt = ttt

    # Calculate train_loss and train_acc at the end of each epoch
    print("-" * 64)
    train_loss = running_train_loss / len(train_loader)
    train_acc = running_train_corrects / len(train_loader.dataset)
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    t1 = time.time()
    print(
        f"Epoch: [{epoch} / {args.epochs}]\tTrain_Loss: {train_loss:.4f}\tTrain_Acc: {running_train_corrects} / {len(train_loader.dataset)} ({(train_acc * 100):.2f}%)\tTime: {(t1 - t0):.2f}s")

    return train_loss


def validate(val_loader, model, criterion, epoch, device, writer, args):
    t0 = time.time()
    print("=" * 64)
    print("Evaluating model predictions on val_set")
    model.eval()
    print(f"Epoch: [{epoch} / {args.epochs}]...")
    running_val_loss = 0
    running_val_corrects = 0
    tt = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward
            out = model(x)
            loss = criterion(out, y)
            _, pred = torch.max(out, 1)
            running_val_loss += loss.item()
            running_val_corrects += torch.sum(pred == y).item()

            # Show logs
            if (i + 1) % 10 == 0:
                ttt = time.time()
                print(
                    f"Epoch: [{epoch} / {args.epochs}]\tStep: [{i + 1} / {len(val_loader)}]\tLoss: {loss.item():.4f}\tTime: {(ttt - tt):.2f}s")
                tt = ttt

        # Calculate val_loss and val_acc at the end of each epoch
        print("-" * 64)
        val_loss = running_val_loss / len(val_loader)
        val_acc = running_val_corrects / len(val_loader.dataset)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        t1 = time.time()
        print(
            f"Epoch: [{epoch} / {args.epochs}]\tVal_Loss: {val_loss:.4f}\tVal_Acc: {running_val_corrects} / {len(val_loader.dataset)} ({(val_acc * 100):.2f}%)\tTime: {(t1 - t0):.2f}s")

    return val_acc


def save_checkpoint(state, checkpoint_dir, model_name, criterion_name, optimizer_name, epoch, train_loss):
    print("=" * 64)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{date}_{model_name}_{criterion_name}_{optimizer_name}_{epoch}_{train_loss:.4f}.tar")
    print(f"Saving model's checkpoint into: {save_path}")
    torch.save(state, save_path)
    torch.save(state, os.path.join(checkpoint_dir, "latest.tar"))


def main():
    start_time = time.time()
    args = parse_args()
    assert args.model in ["PointNet", "VFE", "VFE_LW"], "'model' param must be 'PointNet' or 'VFE' or 'VFE_LW'"

    # Create experiment directory
    model_dir = os.path.join(args.save_dir, args.model)
    assert args.exp not in os.listdir(model_dir), f"Name {args.exp} already exists, please change to another name"
    save_dir = os.path.join(model_dir, args.exp)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Save log to file
    sys.stdout = Logger(filename=f"{save_dir}/logs.txt", stream=sys.stdout)

    print("args: ", args)

    # Set seed
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading train/val datasets and dataloaders
    train_set = KittiClsDataset("../../data/kitti_cls/train")
    val_set = KittiClsDataset("../../data/kitti_cls/val")
    print(f"Train Set: {len(train_set)} samples")
    print(f"Val Set: {len(val_set)} samples")

    # Get weighted random samplers for train dataloader and val dataloader
    train_sampler = get_weighted_random_sampler(train_set)
    val_sampler = get_weighted_random_sampler(val_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)

    # Model
    if args.model == "PointNet":
        model = PointNet().to(device)
    elif args.model == "VFE":
        model = VFE(point_nums=1000).to(device)
    else:
        model = VFE_LW(point_nums=1000).to(device)
    model_name = model.__class__.__name__
    print("Model: ", model_name)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_name = optimizer.__class__.__name__
    print("Optimizer: ", optimizer_name)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr_every, gamma=args.decay_lr_factor)

    # Loss
    criterion = nn.CrossEntropyLoss()
    criterion_name = criterion.__class__.__name__
    print("Loss: ", criterion_name)

    # Setup Tensorboard
    with SummaryWriter(save_dir) as writer:
        start_epoch = 1
        best_val_acc = 0

        print("-" * 64)
        print("Start Training")
        for epoch in range(start_epoch, args.epochs + 1):
            # Train for one epoch
            train_loss = train(train_loader, model, criterion, optimizer, epoch, device, writer, args)

            # Update learning rate
            scheduler.step()

            # Evaluate on validation set after each epoch
            val_acc = validate(val_loader, model, criterion, epoch, device, writer, args)

            # remember best val acc and save checkpoint
            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc, best_val_acc)
            if is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }, checkpoint_dir, model_name, criterion_name, optimizer_name, epoch, train_loss)

    print("#" * 64)
    print("Done Training")
    end_time = time.time()
    print(f"Time spent: {(end_time - start_time):.2f}s")


if __name__ == '__main__':
    main()
