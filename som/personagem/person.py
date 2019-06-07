# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

from  minisom import  MiniSom

from sklearn.preprocessing import MinMaxScaler , LabelEncoder

from pylab import pcolor, colorbar, plot



base = pd.read_csv('personagens.csv')

X = base.iloc[:, :6].values

y = base.iloc[:, 6].values



labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)



normalizador = MinMaxScaler(feature_range=(0,1))

X = normalizador.fit_transform(X)



som  = MiniSom(x = 293, y = 293, input_len = 6, random_seed = 0)

som.random_weights_init(X)

som.train_random(data = X, num_iteration = 100)



pcolor(som.distance_map().T)

colorbar()



markers = ['o', 's']

colors = ['r','g']



for i , x in enumerate(X):

    w = som.winner(x)

    plot(w[0] + 0.5, w[1] + 0.5 , markers[y[i]],

         markerfacecolor = 'None', markersize = 10,

         markeredgecolor= colors[y[i]], markeredgewidth = 2)



mapeamento = som.win_map(X)

suspeitos = np.concatenate((mapeamento[(1,1)],mapeamento[(2,5)]),axis = 0)

suspeitos = normalizador.inverse_transform(suspeitos)

