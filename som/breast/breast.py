'''

learning_rate = 0.01

sigma = 0.9

num_iteration = 200

'''

from minisom import MiniSom

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from pylab import pcolor, colorbar, plot



entradas = pd.read_csv('entradas-breast.csv')

classe = pd.read_csv('saidas-breast.csv')



X = entradas.iloc[:, 0:30].values

Y = classe.iloc[:, 0].values



# normaliza os dados

normalizador = MinMaxScaler(feature_range = (0,1))

X = normalizador.fit_transform(X)



# CONSTRUCAO DO MAPA AUTO ORGANIZAVEL

som = MiniSom(x = 50, y = 50, input_len = 30, sigma = 1, learning_rate = 0.01, random_seed = 2)

# inicializa os pesos na base de dados X

som.random_weights_init(X)

# inicia o treinamento

som.train_random(data = X, num_iteration = 10000, verbose = True)



pcolor(som.distance_map().T)

colorbar()



# VISUALIZACAO DOS RESULTADOS

markers = ['o', 's']

color = ['r', 'g']



for i, x in enumerate(X):

    w = som.winner(x)

    plot(w[0] + 0.5,

         w[1] + 0.5,

         markers[Y[i]],

         markerfacecolor = 'none',

         markersize = 10,

         markeredgecolor = color[Y[i]],

         markeredgewidth = 2            # Borda

         

         )