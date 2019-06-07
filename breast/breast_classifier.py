# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
def criarRede(optimazer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    #otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = optimazer, loss = loss, metrics = ['binary_accuracy'])
    
    return classificador

classificador = criarRede('Adamax', 'binary_crossentropy', 'random_uniform', 'relu', 16)
classificador.fit(previsores, classe, batch_size = 30, epochs = 1000)

#novo = np.array([[ 17.99,	10.38,	122.8,	1001.0,	0.1184	,0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1095.0,	0.9053,	8589.0,	153.4	,0.006399	,0.04904,	0.05372999999999999,	0.01587	,0.03003	,0.006193	,25.38	,17.33	,184.6,	2019.0,	0.1622,	0.6656	,0.7119,	0.2654	,0.4601	,0.1189]])
#previsao = classificador.predict(novo)
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')    

