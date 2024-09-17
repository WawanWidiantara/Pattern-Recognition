import cv2, time

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
id_user = input("Enter the id of the person: ")
a = 0
while True:
    a = a + 1
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in face:
        cv2.imwrite(
            "dataSet/User." + str(id_user) + "." + str(a) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Capturing", frame)
    if a > 49:
        break

video.release()
cv2.destroyAllWindows()
