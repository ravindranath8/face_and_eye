import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load pre-trained classifiers
face_classifier = cv2.CascadeClassifier(r'G:\Data Science\PRAKAS senapati\Regular class\NOV_8_2024_cv2\6th - intro to cv2\Haarcascades\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(r'G:\Data Science\PRAKAS senapati\Regular class\NOV_8_2024_cv2\6th - intro to cv2\Haarcascades\haarcascade_eye.xml')

st.title("Face and Eye Detection App")
st.write("Upload an image, and the app will detect faces and eyes.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","jfif"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        st.write("No face found.")
    else:
        # Draw rectangles around faces and eyes
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (127, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_classifier.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)

        # Display the output image with detections
        st.image(img, caption="Detected faces and eyes", use_column_width=True)


