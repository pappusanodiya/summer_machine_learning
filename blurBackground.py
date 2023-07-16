
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)  

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    blur = cv2.GaussianBlur(frame, (99, 99), 0)

    for (x, y, w, h) in faces:
        
        left = frame[:, 0:x]  
        blur_roi = blur[:, 0:x] 
        frame[:, 0:x] = blur_roi
        
        
        right = frame[:, x+w:]  
        blur_roi = blur[:, x+w:] 
        frame[:, x+w:] = blur_roi
        
        upper = frame[0:y, :]  
        blur_roi = blur[0:y, :] 
        frame[0:y, :] = blur_roi
        
        lower = frame[y+h:, :]  
        blur_roi = blur[y+h:, :] 
        frame[y+h:, :] = blur_roi

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Face Detection with Background Blur', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
video_capture.release()
 
 
