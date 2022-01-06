import sys
import numpy as np
import os
from tensorflow import keras


sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization")
from visualization.config import ConfigFile
from animation import animate
from meshlib import Mesh

sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/model_testing")
from model_test import input_creation, output_extraction
import phoneme_detection as phde

if __name__ == "__main__":
    face_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"
    audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/audio/sa1.wav"

    df = phde.phoneme_csv_creation(audio_path)
    X = input_creation(df)
    print("Input created")

    model = keras.models.load_model("C:/Users/Enzo.Magal/Documents/Enzo2021/models/model_animation.hdf5")
    y = model.predict(X)
    print("Output computed")

    vert_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/prediction"
    files=os.listdir(vert_path)
    for i in range(0,len(files)):
        os.remove(vert_path+'/'+files[i])
    output_extraction(y,vert_path)
    print("Output extracted !")

    cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/fixed_map.yml")
    animate(vert_path, face_path, cfg)