import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

st.title("YOLOv5 Mask Detection WebApp")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)

st.write("Model loaded")

uploaded_file = st.file_uploader("Upload Image :")

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    results = model(img)
    output = np.squeeze(results.render())
    st.image(output, caption='Output Image', use_column_width=True)
else:
    st.warning("Please upload an image!")

st.write("Thank you for using YOLOv5 Mask Detection WebApp.")
