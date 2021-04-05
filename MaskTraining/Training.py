from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import json
import os

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\ezequ\Desktop\UOC\TFG\TFG\Data"
CATEGORIES = ["incorrect_mask", "with_mask", "without_mask"]

annotations_dir = r"C:\Users\ezequ\Desktop\UOC\TFG\TFG\Data\annotations"
images_dir = r"C:\Users\ezequ\Desktop\UOC\TFG\TFG\Data\images"

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

images = []
labels = []

for filename in os.listdir(images_dir):
    try:
        num = filename.split('.')[0]
        class_name = None
        anno = filename + ".json"
        with open(os.path.join(annotations_dir, anno)) as json_file:
            json_data = json.load(json_file)
            no_anno = json_data["NumOfAnno"]
            k = 0
            for i in range(0, no_anno):
                class_nam = json_data['Annotations'][i]['classname']
                if class_nam == 'face_with_mask':
                    class_name = 'with_mask'
                    k = i
                    break
                elif class_nam == 'face_no_mask':
                    class_name = 'without_mask'
                    k = i
                    break
                # elif class_nam == 'face_with_mask_incorrect':
                #     class_name = 'face_with_mask_incorrect'
                #     k = i
                #     break
                else:
                    if class_nam in ['hijab_niqab', 'face_other_covering', "scarf_bandana", "balaclava_ski_mask", "other"]:
                        class_name = 'without_mask'
                    elif class_nam in ["gas_mask", "face_shield", "mask_surgical", "mask_colorful"]:
                        class_name = 'with_mask'
                    elif class_nam == "face_with_mask_incorrect":
                        class_name = 'incorrect_mask'
                        print(num)
                box = json_data['Annotations'][k]['BoundingBox']
                (x1, x2, y1, y2) = box
        if class_name is not None:
            try:
                image = cv2.imread(os.path.join(images_dir, filename))
                img = image[x2:y2, x1:y1]
                img = cv2.resize(img, (224, 224))
                img = img[..., ::-1].astype(np.float32)
                img = preprocess_input(img)
                images.append(img)
                labels.append(class_name)
            except Exception as e:
                print(str(e))
    except Exception as e:
                print(str(e))

# data = []
# labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    print(category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	images.append(image)
    	labels.append(category)


images = np.asarray(images, dtype="float32")
labels = np.array(labels)


# # # perform one-hot encoding on the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# data = np.array(data, dtype="float32")
# labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(images, labels,
                                                  test_size=0.20, stratify=labels, random_state=120)

print("LEN OF trainX", len(trainX))
print("LEN OF trainY", len(trainY))
print("LEN OF testX", len(testX))
print("LEN OF testY", len(testY)) 


# # # construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# # # load the MobileNetV2 network, ensuring the head FC layer sets are
# # # left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# # # construct the head of the model that will be placed on top of the
# # # the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# # # place the head FC model on top of the base model (this will become
# # # the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


# # # loop over all layers in the base model and freeze them so they will
# # # *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

print(model.summary())
print("[INFO] saving model summary as txt file...")
with open("model_summary.txt",'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# # # compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# # # train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# # # make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# # # for each image in the testing set we need to find the index of the
# # # label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# # # show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# # # serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector_model", save_format="h5")
print("[INFO] mask detector model saved...")

""" # # # plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png") """

# pred = model.predict(testX, batch_size=BS)
# pred = np.argmax(pred, axis=1)
# print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))
