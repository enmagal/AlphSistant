import pandas as pd
import random
import csv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf

def split_data(data):
    data['train/test'] = 'train'
    test_indices = random.sample(range(0,data.shape[0]), int((data.shape[0]*20)/100))
    for i in test_indices :
        data.loc[i, 'train/test'] = 'test'
    return data

def data_normalization(y_train, y_test):
    values = []
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            values.append(y_train[i][y])
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            values.append(y_test[i][y])    
    mean = np.mean(values)
    st = np.std(values)
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            y_train[i][y] = (y_train[i][y]-mean)/st
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            y_test[i][y] = (y_test[i][y]-mean)/st
    return (y_train, y_test)

if __name__ == "__main__": 
    data = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2021/VidTIMIT/data_3_1.csv")
    
    data = split_data(data)
    
    X_train = []
    X_test = []

    y_train = []
    y_test = []

    for i in range(1208):
        path = data.loc[i, 'face_mesh_path']
        f = open(path, 'r')
        # lire le contenu du fichier
        r = csv.reader(f, delimiter=" ", quoting=csv.QUOTE_NONNUMERIC)
        face_mesh = list(r)
        flat_list = [item for sublist in face_mesh for item in sublist]
        f.close()
        if data.loc[i, 'train/test'] == 'train':
            y_train.append(flat_list)
            X_train.append(list(data.loc[i, ['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3']]))
        else :
            y_test.append(flat_list)
            X_test.append(list(data.loc[i, ['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3']]))

    #y_train, y_test = data_normalization(y_train, y_test)

    model = Sequential()
    model.add(tf.keras.Input(shape=(None,33)))
    model.add(Dense(3000, activation="tanh"))
    model.add(Dense(3000, activation="tanh"))
    model.add(Dense(3000, activation="tanh"))
    model.add(Dense(1434))
    model.compile(loss="mse", optimizer="adam")
    
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save('../models/model_animation.hdf5')