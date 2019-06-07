import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:7].values
base_valor_maximo = base.iloc[:, 2:3].values
base_valor_open = base.iloc[:, 1:2].values
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)
base_valor_open_normalizada = normalizador.fit_transform(base_valor_open)

days = 180
sizebase = len(base_treinamento_normalizada)

previsores = []
preco_real_high = []
preco_real_open = []
for i in range(days, sizebase):
    previsores.append(base_treinamento_normalizada[i-days:i, 0:6])
    preco_real_high.append(base_valor_maximo_normalizada[i, 0])
    preco_real_open.append(base_valor_open_normalizada[i, 0])
    
previsores, preco_real_high, preco_real_open = np.array(previsores), np.array(preco_real_high), np.array(preco_real_open)
#previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 6))

preco_real = np.column_stack((preco_real_open, preco_real_high))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 2, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 3, batch_size = 32)

base_teste = pd.read_csv('teste.csv')
preco_real_open = base_teste.iloc[:, 1:2].values
preco_real_high = base_teste.iloc[:, 2:3].values

frames = [base, base_teste]
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis = 1)
entradas = base_completa[len(base_completa) - len(base_teste) - days:].values
#entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

entradas_size = len(entradas)

X_teste = []
for i in range(days, entradas_size):
    X_teste.append(entradas[i-days:i, 0:6])
    
    
X_teste = np.array(X_teste)

    
previsoes = regressor.predict(X_teste)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(preco_real)

previsoes = normalizador_previsao.inverse_transform(previsoes)

   
plt.plot(preco_real_open, color = 'red', label = 'Preço abertura real')
plt.plot(preco_real_high, color = 'black', label = 'Preço alta real')

plt.plot(previsoes[:, 0], color = 'blue', label = 'Previsões abertura')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsões alta')

plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()












