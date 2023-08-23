import os
import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def extract_video(video, dict_faces, video_folder_path):
    """
    Crops faces according to the dictionary containing the positions of the faces.
    The extracted faces are collected in the folder of the linked video.
    :param video: File paths of video
    :param dict_faces: Dictionary with the position of the faces in the video
    :param video_folder_path: Folder path to extract faces
    """
    try:
        bboxes_dict = dict_faces
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success or int(i) not in bboxes_dict:
                continue
            crops = []
            bboxes = bboxes_dict[int(i)]
            if bboxes is None:
                continue
            else:
                counter += 1
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = 0
                p_w = 0
                if h > w:
                    p_w = int((h-w)/2)
                elif h < w:
                    p_h = int((w-h)/2)

                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                h, w = crop.shape[:2]
                crops.append(crop)

            for j, crop in enumerate(crops):
                cv2.imwrite(os.path.join(video_folder_path,"{}_{}.png".format(i, j)), crop)
        if counter == 0:
            print(video, counter)
    except e:
        print("Error:", e)