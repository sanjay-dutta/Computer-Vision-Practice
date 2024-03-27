#  Object Detection with Haar Cascades
import cv2

# Load the Haar Cascade for face detection
cascade_path = "D:\Github_Desktop\Computer-vision-practice\Practice\haarcascade_frontalcatface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the image on which to perform object detection
image_path = "D:\Github_Desktop\Computer-vision-practice\Practice\pp.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
