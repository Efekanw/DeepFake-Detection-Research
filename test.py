import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import os
import cv2
import numpy as np
import torch
from functools import partial
from os import cpu_count
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from utils import custom_round, custom_video_round
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
import yaml
import argparse
from models.model import model_instance

#########################
####### CONSTANTS #######
#########################

DATA_DIR = os.path.join("dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_dataset")
OUTPUT_DIR = os.path.join("models")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.txt")


def save_confusion_matrix(confusion_matrix, model_name):
    try:
        matplotlib.use('agg')
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, cmap="Blues")

        threshold = im.norm(confusion_matrix.max()) / 2.
        textcolors = ("black", "white")

        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(["original", "fake"])
        ax.set_yticklabels(["original", "fake"])

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",
                               fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

        fig.tight_layout()
        image_path = os.path.join(OUTPUT_DIR, "confusion.jpg")
        plt.savefig(image_path)
    except Exception as e:
        print(e)


def save_roc_curves(correct_labels, preds, model_name):
    try:
        matplotlib.use('agg')
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')

        fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

        model_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label="Model_" + model_name + ' (area = {:.3f})'.format(model_auc))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        roc_path = os.path.join(OUTPUT_DIR, "roc.jpg")
        plt.savefig(roc_path)
        plt.clf()
    except Exception as e:
        print(e)


def read_frames(video_path, videos, frames_per_video, config):

    test_df = pd.read_csv(TEST_LABELS_PATH, sep=",")
    video_folder_name = os.path.basename(video_path)
    video_key = video_folder_name + ".mp4"
    # label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    print(video_key)
    label = test_df.loc[test_df['filename'] == video_key, 'label'].iloc[0]

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video

    for path in frames_paths:
        for i in range(0, 3):  # Consider up to 3 faces per video
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

            frames_paths_dict[key] = frames_paths_dict[key][:frames_per_video]

    # Select N frames from the collected ones
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    videos.append((video, label, video_path))


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


#########################
#######   MODEL   #######
#########################


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', default=10, type=int,
                         help='Number of data loader workers.')
    parser.add_argument('--model_path', default='models/efficientnet_checkpoint', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--max_videos', type=int, default=-1,
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='configs/architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0,
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30,
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument('--model', type=str, default='cross_efficient_vit',
                        help="Model name")

    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if os.path.exists(opt.model_path):
        model = model_instance(model_name=opt.model, config=config)
        model.load_state_dict(torch.load(opt.model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)

    #########################
    ####### EXECUTION #######
    #########################

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    NUM_CLASSES = 1
    preds = []

    paths = []
    videos = []

    # Read all videos paths
    for index, video_folder in enumerate(os.listdir(TEST_DIR)):
        paths.append(os.path.join(TEST_DIR, video_folder))

    #Read faces
    with Pool(processes=cpu_count()-1) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos, config=config), paths):
                pbar.update()


    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []
    folder_name = model_name.split('.')[0]
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    bar = Bar('Predicting', max=len(videos))

    f = open(OUTPUT_DIR + "/" + "_" + model_name + "_labels.txt", "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), opt.batch_size):
                faces = video_faces[i:i + opt.batch_size]
                faces = torch.tensor(np.asarray(faces))
                if faces.shape[0] == 0:
                    continue
                faces = np.transpose(faces, (0, 3, 1, 2))
                faces = faces.float()

                pred = model(faces)

                scaled_pred = []
                for idx, p in enumerate(pred):
                    scaled_pred.append(torch.sigmoid(p))
                faces_preds.extend(scaled_pred)
            del faces
            del scaled_pred
            current_faces_pred = sum(faces_preds) / len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            f.write(" " + str(face_pred))
            video_faces_preds.append(face_pred)
        del faces_preds
        del current_faces_pred
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])

        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" + "\n")
        del video_faces_preds
    f.close()
    bar.finish()

    #########################
    #######  METRICS  #######
    #########################

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)

    loss = loss_fn(tensor_preds, tensor_labels).numpy()
    loss = loss.item()
    loss = round(loss % 1, 2)
    # accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels) # Classic way
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)  # Custom way
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    result = (model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    with open(OUTPUT_DIR + 'metrics.txt', encoding='utf8', mode='w+') as f:
        f.write(result)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
    save_confusion_matrix(metrics.confusion_matrix(correct_test_labels, custom_round(np.asarray(preds))),
                                      model_name=model_name)