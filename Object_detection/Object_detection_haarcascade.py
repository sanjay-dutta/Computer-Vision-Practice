import cv2

# Load the cascades
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Add path to other cascades for cat, car, pedestrian detection
# Example: haarcascade_frontalcatface.xml, haarcascade_car.xml, haarcascade_fullbody.xml
# You may need to download these XML files if they are not included in your OpenCV installation
#cat_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalcatface.xml')
car_cascade = cv2.CascadeClassifier('D:\Github_Desktop\Computer-vision-practice\Object_detection\haarcascade_car.xml')
#pedestrian_cascade = cv2.CascadeClassifier('path/to/haarcascade_fullbody.xml')

# Function to detect objects
def detect_objects(img, cascade, scale_factor=1.1, min_neighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return objects

cap = cv2.VideoCapture('D:\Github_Desktop\Computer-vision-practice\Object_detection\hd_25fps.mp4')
resize_factor = 0.5  # Factor to resize the video frames for detection

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster detection
    small_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
    
    # Detect objects in the smaller frame
    # faces = detect_objects(small_frame, face_cascade)
    # cats = detect_objects(small_frame, cat_cascade)
    cars = detect_objects(small_frame, car_cascade)
    # pedestrians = detect_objects(small_frame, pedestrian_cascade)
    
    # Draw rectangles around detected objects on the original frame
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (int(x / resize_factor), int(y / resize_factor)),
    #                   (int((x + w) / resize_factor), int((y + h) / resize_factor)), (255, 0, 0), 2)
    # Repeat for cats, cars, pedestrians with different colors if desired
    # Draw rectangles around detected cars on the original frame
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (int(x / resize_factor), int(y / resize_factor)),
                      (int((x + w) / resize_factor), int((y + h) / resize_factor)), (0, 255, 0), 2)
    
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
