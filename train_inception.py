import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D, Conv2D, MaxPool2D, \
    BatchNormalization, InputLayer, GlobalAveragePooling2D
from tensorflow_core.python.keras import regularizers
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.optimizers import SGD, RMSprop
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_DIR = 'train/'
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 299
LABELS = ['cold-arms', 'fight', 'fire', 'firearms', 'gore', 'non-violence']

data = []
labels = []
trainImagesPath = []

print("[INFO] loading images...")
# build list of images
for label in LABELS:
    listDir = os.listdir(TRAIN_DIR + label)
    for item in listDir:
        trainImagesPath.append(TRAIN_DIR + label + "/" + item)

# print(trainImagesPath)
with tqdm(total=len(trainImagesPath)) as pbar:
    for path in trainImagesPath:
        pbar.update(1)
        label = path.split(os.path.sep)[-2]
        if label not in LABELS:
            continue

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        data.append(image)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split data into 75% test and 25% validation batches
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  stratify=labels,
                                                  random_state=42)

# initialize training data augmentation object
print("[INFO] init training data...")
trainAug = ImageDataGenerator(rotation_range=30,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.15,
                              horizontal_flip=True,
                              fill_mode="nearest")
validAug = ImageDataGenerator()

# mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# trainAug.mean = mean
# validAug.mean = mean

# construct the model
print("[INFO] construction model...")
baseModel = InceptionV3(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] compiling model...")
# opt = SGD(lr=1e-4, momentum=0.6, decay=1e-4 / EPOCHS)
opt = RMSprop(lr=1e-5)
model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"],
              optimizer=opt)

# train the head of the network
H = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE, shuffle=True),
                        steps_per_epoch=len(trainX) // BATCH_SIZE,
                        validation_data=validAug.flow(testX, testY, shuffle=True),
                        validation_steps=len(testX) // BATCH_SIZE,
                        epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluation network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# save model to disk
print("[INFO] saving model...")
model.save("violence_inception.model")

f = open("lb_inception.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
