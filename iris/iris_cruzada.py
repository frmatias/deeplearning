# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import model_from_json 

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(neurons = 16):
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dense(units = neurons, activation = 'relu'))
    classificador.add(Dense(units = neurons, activation = 'relu'))
    classificador.add(Dense(units = neurons, activation = 'relu'))
    classificador.add(Dense(units = neurons, activation = 'relu'))
    classificador.add(Dense(units = neurons, activation = 'relu'))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    return classificador

#classificador = KerasClassifier(build_fn = criarRede, epochs = 1000, batch_size =10)

#resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')

#media = resultados.mean()

#devio = resultados.std()

rede = criarRede()
rede.fit(previsores,classe_dummy, epochs = 1000, batch_size =150)
rede_json = rede.to_json()
with open('rede_json.json', 'w') as json_file:
    json_file.write(rede_json)
rede.save_weights('rede_weights.h5')    



