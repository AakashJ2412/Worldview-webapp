import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import fastai.vision
import torch
from fastai.vision import *
from fastai.vision.learner import *
from fastai.vision.interpret import *
from pathlib import Path
from fastai.utils.mem import *
torch.backends.cudnn.benchmark=True
import numpy as np
import os
from model import *


st.title("Worldview Web Application")
st.markdown("This application allows you to explore the Worldview application, and upload the image of your choice to see the results of the model.")
col1, col2 = st.columns(2, gap="large")
uploadedFlag = False
uploadedImage = None
doneProcessing = False

with col1:
    if not uploadedFlag:
        selectedOption = st.selectbox("Select input format", ("Upload Image", "Take Photo"))
        inputFile = None
        if selectedOption == "Upload Image":
            inputFile = st.file_uploader("Upload an image", type=["jpg", "png"])
        elif selectedOption == "Take Photo":
            inputFile = st.camera_input("Take a picture", key="camera")

    if inputFile is not None:
        uploadedFlag = True
        uploadedImage = PIL.Image.open(inputFile).convert('RGB')
        # background = PIL.Image.new('RGBA', uploadedImage.size, (255, 255, 255))
        # uploadedImage = PIL.Image.alpha_composite(background, uploadedImage)

        st.image(uploadedImage, use_column_width=True)
        st.write("Image uploaded successfully")
        st.write("Image size: ", uploadedImage.size)


with col2:
    if uploadedImage is not None:
        if st.button("Process Image"):
            with st.spinner(text="In progress..."):
                runModel(uploadedImage)
            st.success('Done!')
            doneProcessing = True

    if not uploadedFlag:
        st.write("No image uploaded...")

    if doneProcessing:
        st.image("./results/colored.png", use_column_width=True)
