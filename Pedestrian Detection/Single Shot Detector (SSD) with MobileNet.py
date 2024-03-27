import cv2
import tensorflow as tf
import numpy as np

# Correct the path using a raw string or double backslashes
model_path = r'D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\ssd_mobilenet_v2_coco_2018_03_29\saved_model'
model = tf.saved_model.load(model_path)

# Get the serving signature for inference
serve = model.signatures['serving_default']

def run_detection(image):
    # Convert the image to a tensor and expand dimensions to match model's input
    input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
    
    # Perform inference
    detections = serve(tf.constant(input_tensor))['detection_boxes']

    return detections

# Correct the path for the video
video_path = r'D:\Github_Desktop\Computer-vision-practice\Pedestrian Detection\padestrians.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    detections = run_detection(frame_rgb)
    
    # Process detections (assuming detections are normalized coordinates)
    height, width, _ = frame.shape
    for detection in detections[0]:
        ymin, xmin, ymax, xmax = detection.numpy()
        (left, right, top, bottom) = (xmin * width, xmax * width, 
                                      ymin * height, ymax * height)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        
        # Draw detection boxes
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
