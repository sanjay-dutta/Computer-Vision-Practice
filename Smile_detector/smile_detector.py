import cv2
import dlib
import numpy as np

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Github_Desktop\Computer-vision-practice\Smile_detector\shape_predictor_68_face_landmarks.dat")

def smile_distance(shape):
    """
    Calculate the normalized smile distance.
    """
    left = np.array([shape.part(48).x, shape.part(48).y])  # Left corner of the mouth
    right = np.array([shape.part(54).x, shape.part(54).y])  # Right corner of the mouth
    top_nose = np.array([shape.part(27).x, shape.part(27).y])  # Top of the nose for reference
    smile_width = np.linalg.norm(left - right)
    face_width = np.linalg.norm(top_nose - left) + np.linalg.norm(top_nose - right)
    normalized_smile_distance = smile_width / face_width
    return normalized_smile_distance

cap = cv2.VideoCapture(0)  # Start the webcam

while True:
    _, frame = cap.read()
    if frame is None:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        smile_dist = smile_distance(landmarks)

        # Display the smile distance on the frame
        cv2.putText(frame, f"Smile: {smile_dist:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if smile_dist > 0.25:  # Threshold for smile detection, adjust based on your testing
            print("Smile Detected!")
            cv2.imwrite("smile.jpg", frame)  # Save the frame as an image
            cv2.putText(frame, "Smile Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smile Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
