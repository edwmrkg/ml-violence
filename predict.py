import pickle
import argparse
import cv2
import numpy as np

from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-l", "--label_bin", default="lb.pickle", help="path to label binarizer")
ap.add_argument("-m", "--model", default="violence.model", help="Path to model")
ap.add_argument("-s", "--size", default=299, help="Image size")
args = vars(ap.parse_args())

model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
IMAGE_SIZE = args["size"]

image = cv2.imread(args["input"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype("float32")
image -= mean

prediction = model.predict(np.expand_dims(image, axis=0))[0]
label = lb.classes_[np.argmax(prediction)]

for i in range(len(prediction)):
    print(lb.classes_[i] + ":\t", prediction[i])

print("Result: ", label, " [%s]" % prediction[np.argmax(prediction)])


