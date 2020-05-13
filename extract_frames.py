import pandas as pd
import cv2
import math


def getDataFrame(file):
    # read file
    f = open(file)
    temp = f.read()
    videos = temp.split('\n')

    # create data frame
    video_data = pd.DataFrame()
    video_data['name'] = videos
    video_data = video_data[:-1]
    video_data.head()

    # create tags
    video_set_tag = []
    for i in range(video_data.shape[0]):
        video_set_tag.append(video_data['name'][1].split('/')[1])

    video_data['tag'] = video_set_tag
    return video_data


def extractFrames(files, directory):
    dataframe = getDataFrame(files)

    for i in range(dataframe.shape[0]):
        video_file = dataframe['name'][i]
        extractFramesFromVideo(video_file, directory)


def extractFramesFromVideo(video, directory):
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(5)
    count = 0

    while cap.isOpened():
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % math.floor(frame_rate) == 0:
            file_name = directory + video.split('/')[2] + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(file_name, frame)

    cap.release()
