import streamlit as st
import time



st.header("AlphSistant Demo")

uploaded_file = st.file_uploader("Choose an audio file")
if uploaded_file is not None:

    import sys
    import numpy as np
    import os
    from tensorflow import keras


    sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization")
    from animation import animate
    from meshlib import Mesh

    sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/model_testing")
    from model_test import input_creation, output_extraction
    import phoneme_detection as phde

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.success("Library Imported")

    face_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"
    audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/audio/sa1.wav"

    df = phde.phoneme_csv_creation(audio_path)
    X = input_creation(df)

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.success("Input Created")

    model = keras.models.load_model("C:/Users/Enzo.Magal/Documents/Enzo2021/models/model_animation.hdf5")
    y = model.predict(X)
    print("Output computed")

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.success("Output Computed")

    vert_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/prediction"
    files=os.listdir(vert_path)
    for i in range(0,len(files)):
        os.remove(vert_path+'/'+files[i])
    output_extraction(y,vert_path)
    print("Output extracted !")

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.success("Output Extracted")

    animate(vert_path, face_path)