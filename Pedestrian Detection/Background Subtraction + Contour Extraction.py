import cv2
import numpy as np

# Initialize the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Read the video
cap = cv2.VideoCapture('D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\padestrians.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw contours
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
