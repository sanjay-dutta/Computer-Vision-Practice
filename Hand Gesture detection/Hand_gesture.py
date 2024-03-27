import cv2
import numpy as np

def capture_background(cap, wait=60):
    """
    Capture the background frame.
    """
    for i in range(wait):
        ret, background = cap.read()
        if not ret:
            continue
    # Convert the background to grayscale for easier processing
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    return background

def preprocess_frame(frame, background):
    """
    Preprocess the frame for hand segmentation using background subtraction.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(background, gray_frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return thresh

def count_fingers(thresh, frame):
    """
    Count the fingers in the segmented hand region.
    """
    # Initialize finger_count at the start of the function
    finger_count = 0
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assuming the largest external contour in frame is the hand
        hand_contour = max(contours, key=cv2.contourArea)
        # Convex hull around the hand
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(hand_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):  # Calculate the angle
                        s, e, f, d = defects[i][0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(far) - np.array(end))
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / np.pi)
                        # If the angle < 90 degrees, consider it as a finger
                        if angle <= 90:
                            finger_count += 1
            except:
                pass  # Handle errors related to convexity defects calculation
            
    return finger_count




def main():
    cap = cv2.VideoCapture(0)
    cv2.waitKey(1000)  # Wait for the camera to initialize
    background = capture_background(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thresh = preprocess_frame(frame, background)
        finger_count = count_fingers(thresh, frame)

        # Display the finger count on the frame
        if finger_count is not None:
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Thresh', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
