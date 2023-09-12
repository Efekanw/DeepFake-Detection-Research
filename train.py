import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from models.ViT.cross_efficient_vit import CrossEfficientViT
from models.model import model
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse

DATA_DIR = os.path.join("dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "train_dataset")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_dataset")
MODELS_PATH = "models"
TRAINING_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.txt")
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "val_labels.txt")


def read_frames(video_path, train_dataset, validation_dataset):
    # Get the video label based on dataset selected
    print(TRAINING_DIR)
    print(video_path)
    if TRAINING_DIR in video_path:
        train_df = pd.read_csv(TRAINING_LABELS_PATH, sep=",")
        video_folder_name = os.path.basename(video_path)
        video_folder_name += '.mp4'

        print(video_folder_name)
        print(train_df['filename'])
        print("TTT" + str(train_df['filename'] == video_folder_name))

        label = train_df.loc[train_df['filename'] == video_folder_name]['label'].values[0]
        print(label)
    else:
        val_df = pd.read_csv(VALIDATION_LABELS_PATH, sep=",")
        video_folder_name = os.path.basename(video_path)
        video_folder_name += '.mp4'
        print(video_folder_name)
        print(val_df.loc[val_df['filename'] == video_folder_name]['label'])
        label = val_df.loc[val_df['filename'] == video_folder_name]['label'].values[0]

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),
                               1)  # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']), 1)

    if VALIDATION_DIR in video_path:
        min_video_frames = int(max(min_video_frames / 8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0, 1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]

            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            # image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=3, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--config', type=str, default='configs/architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--patience', type=int, default=5,
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--model', type=str, default='cross_efficient_vit',
                        help="Model name")

    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # DİNAMİK
    model = model(opt.model, config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'],
                                weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'],
                                    gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    for dataset in sets:
        subfolder = os.path.join(dataset, 'crops')
        if os.path.isdir(os.path.join(subfolder)):
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                video_folder = video_folder_name.split('.')[0]
                if os.path.isdir(os.path.join(subfolder, video_folder)):
                    paths.append(os.path.join(subfolder, video_folder))

    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()

    with Pool(processes=4) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset,
                                              validation_dataset=validation_dataset), paths):
                pbar.update()

    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)

    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)

    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset], dtype=object), labels,
                                     config['model']['image-size'])
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                     batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                     pin_memory=False, drop_last=False, timeout=0,
                                     worker_init_fn=None, prefetch_factor=2,
                                     persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset], dtype=object),
                                          validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True,
                                         sampler=None,
                                         batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                         pin_memory=False, drop_last=False, timeout=0,
                                         worker_init_fn=None, prefetch_factor=2,
                                         persistent_workers=False)
    del validation_dataset

    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0

        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl) * config['training']['bs']) + len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            labels = labels.float()
            y_pred = model(images)
            y_pred = y_pred.cpu()
            print("y_pred" + str(y_pred))
            print("labels" + str(labels))

            loss = loss_fn(y_pred, labels)

            corrects, positive_class, negative_class = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(config['training']['bs']):
                bar.next()

            if index % 1200 == 0:
                print("\nLoss: ", total_loss / counter, "Accuracy: ",
                      train_correct / (counter * config['training']['bs']), "Train 0s: ", negative, "Train 1s:",
                      positive)

        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0

        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
            val_images = np.transpose(val_images, (0, 3, 1, 2))

            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_labels = val_labels.float()
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            bar.next()

        scheduler.step()
        bar.finish()

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
              str(total_loss) + " accuracy:" + str(train_correct) + " val_loss:" + str(
            total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(
            np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(
            np.count_nonzero(validation_labels == 1)))

        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH, "efficientnet_checkpoint" + str(t)))