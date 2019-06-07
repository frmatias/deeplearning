# -*- coding: utf-8 -*-

import pandas as pd
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
classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10],
        'epochs':[500],
        'optimazer':['adam','Adamax'],
        'loss':['binary_crossentropy'],
        'kernel_initializer':['random_uniform'],
        'activation':['relu'],
        'neurons':[16]}

grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

