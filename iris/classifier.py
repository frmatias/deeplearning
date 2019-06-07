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

arquivo = open('rede_json.json', 'r')

estrutura_rede = arquivo.read()

arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('rede_weights.h5')

#novo = np.array([[ 17.99,	10.38,	122.8,	1001.0,	0.1184	,0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1095.0,	0.9053,	8589.0,	153.4	,0.006399	,0.04904,	0.05372999999999999,	0.01587	,0.03003	,0.006193	,25.38	,17.33	,184.6,	2019.0,	0.1622,	0.6656	,0.7119,	0.2654	,0.4601	,0.1189]])
#previsao = classificador.predict(novo)
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#resultado = classificador.evaluate(previsores, classe_dummy)
novo = np.array([[5.1,3.5,1.4,0.2]])
previ = classificador.predict(novo)
previ2 = [np.argmax(t) for t in previ]
