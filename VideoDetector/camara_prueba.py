# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input;
from tensorflow.keras.preprocessing.image import img_to_array;
from tensorflow.keras.models import load_model;
from imutils.video import VideoStream;
import os;
from os.path import join;
import pygame;
import numpy as np;
import imutils;
import time;
import cv2;


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load alarm
pygame.init()
detected_sound = pygame.mixer.Sound(r"alarm.WAV")

# load our serialized face detector model from disk
prototxtPath = join(r"..", r"deploy.prototxt")
weightsPath = join(r"..", r"res10_300x300_ssd_iter_140000.caffemodel")

faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# load the face mask detector model from disk
maskNet = load_model(join(r"..", r"mask_detector_model_finetun_2dataset"))
#maskNet = load_model(join(r"..", r"mask_detector_model2dataset"))
# maskNet = load_model(r"C:\Users\ezequ\Desktop\UOC\TFG\TFG\mask_detector_model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:

    cv2.ocl.setUseOpenCL(False)
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 720 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=720)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (face_incorrect_mask, face_with_mask, face_no_mask) = pred
        print(round(face_incorrect_mask,3),round(face_with_mask,3),round(face_no_mask,3))

        if face_with_mask > (face_no_mask + face_incorrect_mask):
            label = "Mask"
        elif face_no_mask > (face_with_mask + face_incorrect_mask):
            label = "No Mask"
        else:
            label = "Incorrect Mask"

        if label == "Mask":
            color = (0, 255, 0)
        elif label == "No Mask":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(
            face_no_mask, face_with_mask, face_incorrect_mask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # check if the labels is not mask, play an alarm
        # loop = 0 only play once, maxtime = 50 ms its maxtime to stop the sound
        if label.split(":")[0] != "Mask":
            detected_sound.play(loops=0, maxtime=50)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
