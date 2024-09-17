import cv2, time
import os
from PIL import Image

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("dataset/training.xml")

a = 0
while True:
    a = a + 1
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        Id, conf = recognizer.predict(gray[y : y + h, x : x + w])
        if Id == 1:
            Id = "SmartUser"
        cv2.putText(
            frame,
            str(Id),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
