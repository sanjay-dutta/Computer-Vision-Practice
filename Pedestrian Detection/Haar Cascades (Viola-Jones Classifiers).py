import cv2

# Load the Haar cascade file for pedestrian detection
cascade = cv2.CascadeClassifier('D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\haarcascade_frontalface_default.xml')

# Read the video
cap = cv2.VideoCapture('D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\padestrians.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
