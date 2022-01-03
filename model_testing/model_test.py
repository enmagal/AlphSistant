import pandas as pd
import numpy as np

from tensorflow import keras

import phoneme_detection as phde

def input_creation(phoneme_dataframe):
    df_frame =  pd.DataFrame(0, index = [], columns=['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3'])

    for i in range(len(phoneme_dataframe)-3):
        for y in range(3):
            df_frame.loc[i, ['a'+ str(y+1), 'o'+ str(y+1), 'c'+ str(y+1), 'g'+ str(y+1), 'l'+ str(y+1), 'b'+ str(y+1), 'f'+ str(y+1), 't'+ str(y+1), 'j'+ str(y+1), 'u'+ str(y+1), 'q'+ str(y+1)]] = list(phoneme_dataframe.loc[i+y, ['a', 'o', 'c', 'g', 'l', 'b', 'f', 't', 'j', 'u', 'q']])
    X = []
    for i in range(len(df_frame)):
        X.append(list(df_frame.loc[i, ['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3']]))
    
    return(X)

def output_extraction(y, save_path):
    output = []
    for i in range(len(y)):
        newlist = []
        for v in range(0, len(y[i]), 3):
            sublist = []
            for z in range(3):
                sublist.append(y[i][v+z])
            newlist.append(sublist)
        output.append(newlist)
    for i in range(len(output)):
        with open(save_path + "/face_" + '{:03}'.format(i) + ".txt", 'w') as f:
            for y in range(len(output[i])):
                f.write('{:.18e}'.format(output[i][y][0]) + ' ')
                f.write('{:.18e}'.format(output[i][y][1]) + ' ')
                f.write('{:.18e}'.format(output[i][y][2]))
                f.write('\n')
    print("Extraction DONE")

if __name__ == "__main__": 
    audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/VidTIMIT/fadg0/audio/sa2.wav"
    df = phde.phoneme_csv_creation(audio_path)
    X = input_creation(df)
    print("Input created")
    model = keras.models.load_model('model_animation.hdf5')
    print("Loaded model from disk")
    y = model.predict(X)
    print("Output computed")
    save_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/VidTIMIT/results/sa2_test/"
    output_extraction(y,save_path)
