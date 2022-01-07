import streamlit as st
import time

import sys
import numpy as np
import os
import torch


sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization")
from visualization.config import ConfigFile
from animation import animate
from meshlib import Mesh

sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/model_testing")
from model_test import input_creation, output_extraction
import phoneme_detection as phde

st.header("AlphSistant Demo")

st.sidebar.header('Fill in the data :')
model_name = st.sidebar.selectbox(
     'Which Model do you want to use ?',
     ('Original Model', 'Shape Keys Model', 'Fancy Model'))

uploaded_file = st.sidebar.file_uploader("Choose an audio file")

if uploaded_file is not None:
    if st.sidebar.button('Test the model') :
        with st.spinner('Wait for it...'):
            time.sleep(5)
            face_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"
            audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/audio/sa1.wav"

            df = phde.phoneme_csv_creation(audio_path)
            X = input_creation(df)

        st.success("Input Created")
        with st.spinner('Wait for it...'):
            time.sleep(5)
            model = torch.load('C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')
            y = model(X)

        st.success("Prediction Computed")

        with st.spinner('Wait for it...'):
            time.sleep(5)
            vert_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/prediction"
            files=os.listdir(vert_path)
            for i in range(0,len(files)):
                os.remove(vert_path+'/'+files[i])
            output_extraction(y,vert_path)

        st.success("Output Extracted")
        st.balloons()

        cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/alphsistant_code/deformation_external/models/lowpoly/markers-cat-voxel.yml")
        animate(vert_path, face_path, cfg)