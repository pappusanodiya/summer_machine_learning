import cv2

KNOWN_FACE_WIDTH = 15.0  
FOCAL_LENGTH = 700.0     

# Load the Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if needed

while True:
    # Read the video frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to gray scale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
       

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Estimate the distance to the face
        face_width_pixels = w
        distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels

        # Display the distance on the frame
        distance_text = f"Distance: {distance:.2f} cm"
        cv2.putText(frame, distance_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection with Background Blur', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
