import cv2
from deepface import DeepFace
import streamlit as st

# Start webcam video capture
cap = cv2.VideoCapture(0)

st.title("Facial Emotion Detection App")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces (replace with more robust face detection if needed)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        results = DeepFace.analyze(face_roi, actions=['emotion'])

        # Display emotion on the frame
        cv2.putText(frame, results['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    st.image(frame, channels="BGR")

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
