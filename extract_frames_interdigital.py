import os
from tqdm import tqdm
import cv2

FOLDER = "YouTube-gen/"
ANNOTATIONS_FOLDER = "YouTube-gen/annotations/"
VIDEOS_FOLDER = "YouTube-gen/videos/"
FRAMES_FOLDER = "YouTube-gen/frames/"
FRAMES_PER_VIDEO = 20
FRAMES_DROP = 124

# get files
annotation_files = []
for root, dirs, files in os.walk(ANNOTATIONS_FOLDER):
    if root == ANNOTATIONS_FOLDER:
        annotation_files.extend(files)
annotation_files.sort()

video_files = []
for root, dirs, files in os.walk(VIDEOS_FOLDER):
    if root == VIDEOS_FOLDER:
        video_files.extend(files)
video_files.sort()

# extract annotated frames from videos
for i in tqdm(range(len(video_files))):
    file_name = annotation_files[i].split('.')[0]
    ann_file_name = annotation_files[i]
    vid_file_name = video_files[i]

    # create directory to store frames
    try:
        os.mkdir(FRAMES_FOLDER + file_name)
    except FileExistsError:
        print(FRAMES_FOLDER + file_name + ' exists')

    # read annotation file
    f = open(ANNOTATIONS_FOLDER + ann_file_name, 'r')
    x = f.read().splitlines()
    f.close()

    # parse frames from annotation file
    annotations = []
    for bucket in x:
        words = bucket.split(' ')
        annotations.append([int(words[0]), int(words[1])])

    count = FRAMES_PER_VIDEO
    ann_index = 0
    cap = cv2.VideoCapture(VIDEOS_FOLDER + vid_file_name)
    for frame_id in tqdm(range(int(cap.get(7))), desc=vid_file_name):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAMES_DROP != 0:
            continue

        if ann_index < len(annotations):
            while ann_index < len(annotations) and annotations[ann_index][1] < frame_id:
                ann_index += 1
            if ann_index < len(annotations):
                if annotations[ann_index][0] <= frame_id <= annotations[ann_index][1]:
                    continue

        cv2.imwrite(FRAMES_FOLDER + file_name + '/' + file_name + '_frame%d.jpg' % frame_id, frame)
        count -= 1
        if count <= 0:
            break
    cap.release()
