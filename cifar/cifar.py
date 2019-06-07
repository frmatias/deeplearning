# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.preprocessing.image import image
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

plt.imshow(X_treinamento[2])

plt.title('Classe' + str(y_treinamento[2]))

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               32, 32, 3
                                               )
previsores_teste = X_teste.reshape(X_teste.shape[0],32, 32, 3)

previsores_treinamento = previsores_treinamento.astype('float32')

previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255 

previsores_teste /= 255
 
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)

classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (32, 32, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu' ))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu' ))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax' ))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size =128, epochs = 5, validation_data = (previsores_teste, classe_teste))

image_teste = image.load_img('cat1.png')
image_teste = image.img_to_array(image_teste)
image_teste /= 255
image_teste = np.expand_dims(image_teste, axis = 0)
previsao = classificador.predict(image_teste)
#previsao = [np.argmax(t) for t in previsao]
print('airplane: ', round(previsao[0][0] * 100,2),'%')
print('automobile: ', round(previsao[0][1] * 100,2),'%')
print('bird: ', round(previsao[0][2] * 100,2),'%')
print('cat: ', round(previsao[0][3] * 100,2),'%')
print('deer: ', round(previsao[0][4] * 100,2),'%')
print('dog: ', round(previsao[0][5] * 100,2),'%')
print('frog: ', round(previsao[0][6] * 100,2),'%')
print('horse: ', round(previsao[0][7] * 100,2),'%')
print('ship: ', round(previsao[0][8] * 100,2),'%')
print('truck: ', round(previsao[0][9] * 100,2),'%')
from keras.models import model_from_json 

arquivo = open('classificador.json', 'r')

estrutura_rede = arquivo.read()

arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador.h5')

#novo = np.array([[ 17.99,	10.38,	122.8,	1001.0,	0.1184	,0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1095.0,	0.9053,	8589.0,	153.4	,0.006399	,0.04904,	0.05372999999999999,	0.01587	,0.03003	,0.006193	,25.38	,17.33	,184.6,	2019.0,	0.1622,	0.6656	,0.7119,	0.2654	,0.4601	,0.1189]])
#previsao = classificador.predict(novo)
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

image_teste = image.load_img('cat1.png')
image_teste = image.img_to_array(image_teste)
image_teste /= 255
image_teste = np.expand_dims(image_teste, axis = 0)
previsao = classificador.predict(image_teste)
#previsao = [np.argmax(t) for t in previsao]
print('airplane: ', round(previsao[0][0] * 100,2),'%')
print('automobile: ', round(previsao[0][1] * 100,2),'%')
print('bird: ', round(previsao[0][2] * 100,2),'%')
print('cat: ', round(previsao[0][3] * 100,2),'%')
print('deer: ', round(previsao[0][4] * 100,2),'%')
print('dog: ', round(previsao[0][5] * 100,2),'%')
print('frog: ', round(previsao[0][6] * 100,2),'%')
print('horse: ', round(previsao[0][7] * 100,2),'%')
print('ship: ', round(previsao[0][8] * 100,2),'%')
print('truck: ', round(previsao[0][9] * 100,2),'%')

plt.imshow(image_teste)