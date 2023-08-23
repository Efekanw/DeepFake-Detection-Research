import os
import torch.cuda
import argparse
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import face_detector
from face_detector import VideoDataset
from extract_crops import extract_video

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_videos(videos):
    """
    Detects faces in video frames.
    Gets the position information of the faces into a dictionary and calls extract_video function to crop the faces
    :param videos: File paths of videos
    """
    detector_cls = "FacenetDetector"
    detector = face_detector.__dict__[detector_cls](device=device)
    dataset = VideoDataset(videos)

    loader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []

    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        video_name = os.path.splitext(os.path.basename(video))[0]
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
        for j, frames in enumerate(batches):
            result.update(
                {int(j * detector._batch_size) + i: b for i, b in zip(indices, detector._detect_faces(frames))})
        print(len(result))
        if len(result) > 0:
            video_folder_path = os.path.join(os.path.dirname(video), "crops", video_name)
            os.makedirs(os.path.join(video_folder_path), exist_ok=True)
            extract_video(video=video, dict_faces=result, video_folder_path=video_folder_path)
        else:
            missed_videos.append(id)
    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(id)
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Videos directory')
    opt = parser.parse_args()

    videos_paths = []
    os.makedirs(os.path.join(opt.data_path, "train_dataset"), exist_ok=True)
    os.makedirs(os.path.join(opt.data_path, "test_dataset"), exist_ok=True)
    os.makedirs(os.path.join(opt.data_path, "validation_dataset"), exist_ok=True)

    for folder in os.listdir(opt.data_path):
        for video in os.listdir(os.path.join(opt.data_path, folder)):
            if video[-4:] != '.mp4':
                continue
            else:
                if os.path.isdir(os.path.join(opt.data_path, folder, "crops", video[:-4])):
                    # if folder exists, video processed
                        continue
                videos_paths.append(os.path.join(opt.data_path, folder, video))
    process_videos(videos_paths)


if __name__ == "__main__":
    main()