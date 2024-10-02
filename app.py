import streamlit as st
import torch
import numpy as np
import cv2
from ultralytics import YOLO

st.title("YOLOv5/YOLOv8 Mask Detection WebApp")

# Load YOLOv8 model
model = YOLO('yolov5/runs/train/exp/weights/last.pt')

st.write("Model loaded")

uploaded_file = st.file_uploader("Upload Image:")

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Perform inference
    results = model(img)

    # Render the results on the image
    output = np.squeeze(results[0].plot())  # YOLOv8 uses plot() for rendering
    st.image(output, caption='Output Image', use_column_width=True)
else:
    st.warning("Please upload an image!")

st.write("Thank you for using YOLO Mask Detection WebApp.")
