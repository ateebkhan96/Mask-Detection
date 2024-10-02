import streamlit as st
import torch
import numpy as np
import cv2

st.title("YOLOv8 Mask Detection WebApp")

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True,trust_repo=True)

st.write("Model loaded")

uploaded_file = st.file_uploader("Upload Image:")

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Perform inference using YOLOv8 model
    results = model(img)

    # Render results on the image
    output = results[0].plot()  # Use YOLOv8's plot method
    st.image(output, caption='Output Image', use_column_width=True)
else:
    st.warning("Please upload an image!")

st.write("Thank you for using YOLOv8 Mask Detection WebApp.")
