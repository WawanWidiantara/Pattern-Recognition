import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImage(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert("L")
        faceNp = np.array(faceImg, "uint8")
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        face = detector.detectMultiScale(faceNp)
        for x, y, w, h in face:
            faces.append(faceNp[y : y + h, x : x + w])
            IDs.append(ID)
    return faces, IDs


faces, Ids = getImage("DataSet")
recognizer.train(faces, np.array(Ids))
recognizer.save("dataset/training.xml")
