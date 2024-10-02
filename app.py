import streamlit as st
import torch
import numpy as np
import cv2

st.title("YOLOv5 Mask Detection WebApp")

# Load YOLOv5 model using torch hub (YOLOv5-specific)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)

st.write("Model loaded")

uploaded_file = st.file_uploader("Upload Image:")

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Perform inference using YOLOv5 model
    results = model(img)

    # Render results on the image
    output = np.squeeze(results.render())  # Use YOLOv5's render method
    st.image(output, caption='Output Image', use_column_width=True)
else:
    st.warning("Please upload an image!")

st.write("Thank you for using YOLOv5 Mask Detection WebApp.")
