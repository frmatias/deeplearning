import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras 
from keras.models import Sequential 
from keras.layers import Dense

sel = Sequential()
sel.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
sel.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
sel.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

sel.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

sel.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)
#peso0 = sel.layers[0].get_weights()
#print(peso0)
#previ = sel.predict(previsores_teste)
#previsoes = (previ > 0.5)

#from sklearn.metrics import confusion_matrix, accuracy_score

#precisao = accuracy_score(classe_teste, previsoes)

#matriz = confusion_matrix(classe_teste, previsoes)

resultado = sel.evaluate(previsores_teste, classe_teste)

media = resultado.mean()
