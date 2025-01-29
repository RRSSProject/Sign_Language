import cv2
import os

# Configuration
GESTURES = ["scroll_up", "select"]
IMAGES_PER_GESTURE = 3000
SAVE_DIR = r"R:\Sign_language\Sign-language-recognition\demo" # update the path
IMG_SIZE = (224, 224)

def initialize_directories(gestures, save_dir):
    for gesture in gestures:
        gesture_dir = os.path.join(save_dir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
def wait_for_start_signal(gesture):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam.")
            return False
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Press 's' to start capturing {gesture.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            return True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

def capture_images(gesture, save_dir):
    count = 0
    while count < IMAGES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam.")
            break
        frame = cv2.flip(frame, 1)
        
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        resized_roi = cv2.resize(roi, IMG_SIZE)
        file_path = os.path.join(save_dir, gesture, f"{gesture}{count}.jpg")

        if cv2.imwrite(file_path, resized_roi):
            count += 1
        else:
            print(f"Failed to save: {file_path}")

        progress = (count / IMAGES_PER_GESTURE) * 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Capturing {gesture.upper()} ({count}/{IMAGES_PER_GESTURE})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress:.2f}%", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting early...")
            break
    print(f"Captured {count} images for gesture: {gesture.upper()}")

# Main script
initialize_directories(GESTURES, SAVE_DIR)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

for gesture in GESTURES:
    print(f"\nStarting image capture for gesture: {gesture.upper()}")
    print("Press 's' to start capturing or 'q' to quit.")
    if wait_for_start_signal(gesture):
        capture_images(gesture, SAVE_DIR)
    else:
        print("Skipping this gesture.")
        break

print("Image capture completed. Closing...")
cap.release()
cv2.destroyAllWindows()
