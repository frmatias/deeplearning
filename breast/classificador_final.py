# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.models import model_from_json 

arquivo = open('classificador_breast.json', 'r')

estrutura_rede = arquivo.read()

arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

#novo = np.array([[ 17.99,	10.38,	122.8,	1001.0,	0.1184	,0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1095.0,	0.9053,	8589.0,	153.4	,0.006399	,0.04904,	0.05372999999999999,	0.01587	,0.03003	,0.006193	,25.38	,17.33	,184.6,	2019.0,	0.1622,	0.6656	,0.7119,	0.2654	,0.4601	,0.1189]])
#previsao = classificador.predict(novo)
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
classificador.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)