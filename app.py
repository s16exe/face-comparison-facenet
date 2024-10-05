import os
import cv2
from datetime import datetime, timedelta


# Function to capture face images
def capture_face_images(name):
    # Open video capture
    cap = cv2.VideoCapture(0)

    # Create directory for storing images
    os.makedirs(f'dataset', exist_ok=True)

    start_time = datetime.now()

    count = 0
     # Flag to indicate if we are in capturing mode
    capturing = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now()
        elapsed_time = current_time - start_time



        if not capturing:
            if elapsed_time > timedelta(seconds=5):
                capturing = True
            else:
                print(elapsed_time)

        # Capture images if in capturing mode
        if capturing:
            # Capture image
            count += 1
            cv2.imwrite(f'dataset/{name}2.jpg', frame)
            print(f'Image {count} captured!')

            # Stop capturing after 30 images
            if count >= 1:
                break

        # Show frame
        cv2.imshow('Capturing Faces', frame)

        # Break on 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_face_images("sa")