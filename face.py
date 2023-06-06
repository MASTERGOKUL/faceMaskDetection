import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Adjust threshold value in the range of 80 to 105 based on your lighting conditions.
bw_threshold = 80

# User message
weared_mask = "Thank You for wearing a mask"
not_weared_mask = "Please wear a mask to defeat Corona"

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image to black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if len(faces) == 0 and len(faces_bw) == 0:
        st.write("No face found...")
    elif len(faces) == 0 and len(faces_bw) == 1:
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        st.write(weared_mask)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect mouth contours
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        if len(mouth_rects) == 0:
            st.write(weared_mask)
        else:
            for (mx, my, mw, mh) in mouth_rects:
                if y < my < y + h:
                    # Face and Lips are detected, but lips coordinates are within face coordinates which means lips prediction is true and
                    # person is not wearing a mask
                    st.write(not_weared_mask)
                    break

    # Display the image with bounding boxes
    st.image(img, channels="BGR")
