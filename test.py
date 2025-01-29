import cv2

# Step 1: Create a VideoCapture object
# Use '0' for the default camera, or specify the index or video source path
video_capture = cv2.VideoCapture(0)

# Step 2: Check if the video capture is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video source.")
else:
    # Step 3: Loop to capture frames
    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()
        print(ret, frame)

        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Step 4: Display the frame
        cv2.imshow('Video Feed', frame)

        # Step 5: Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 6: Release resources
    video_capture.release()
    cv2.destroyAllWindows()
