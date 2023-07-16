import cv2
import numpy as np

# Load the pre-trained Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and display cropped faces at the bottom right
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face region to a fixed size
        face_roi = cv2.resize(face_roi, (100, 100))

        # Display the cropped face at the bottom right of the frame
        frame[frame.shape[0]-100:frame.shape[0], frame.shape[1]-100:frame.shape[1]] = face_roi

        # Draw rectangles around the detected faces in the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame with detected faces and cropped faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
