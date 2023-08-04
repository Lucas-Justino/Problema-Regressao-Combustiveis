import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from pylab import plot, show

"""Após selecionar a base de dados, fazemos a leitura, e a guardamos na variavel combustivel. Depois disso, selecionamos quais as colunas(features) que são importantes para o modelo, e excluíamos as features desnecessárias.
Guardamos os valores das features na variável x_combustivel, e os valores da saída desejada na variável y_combustivel
"""

features = ['distance',	'speed',	'temp_inside',	'temp_outside',	'AC', 'rain'] # SELECIONA AS FEATURES IMPORTANTES PARA O MODELO

combustivel.dropna(subset=features, inplace=True) # EXCLUI AS FEATURES DESNECESSÁRIAS

x_combustivel = combustivel[features] # ATRIBUI OS VALORES DAS FEATURES

y_combustivel = combustivel['consume'] # ATRIBUI A SAÍDA DESEJADA

# DIVIDE OS DADOS PARA TREINO E TESTE
x_combustivel_train, x_combustivel_test, y_combustivel_train, y_combustivel_test = train_test_split(x_combustivel, y_combustivel, train_size = 0.75, test_size=0.25, random_state=123)

"""Apresentamos a quantidade de dados em cada variável, onde os dados foram divididos em treino e teste.

"""

print('Tamanho de x_combustivel_train: ', x_combustivel_train.shape) # QUANTIDADE DE DADOS DO X COMBUSTIVEL TRAIN
print('Tamanho de x_combustivel_test: ', x_combustivel_test.shape)   # QUANTIDADE DE DADOS DO X COMBUSTIVEL TESTE
print('Tamanho de y_combustivel_train: ', y_combustivel_train.shape) # QUANTIDADE DE DADOS DO Y COMBUSTIVEL TRAIN
print('Tamanho de y_combustivel_test: ', y_combustivel_test.shape)   # QUANTIDADE DE DADOS DO Y COMBUSTIVEL TRAIN

"""Salvamos e treinamos o regressor"""

from sklearn.tree import DecisionTreeRegressor # IMPORTA O REGRESSOR DA ÁRVORE DE DECISÃO

rgr = DecisionTreeRegressor(max_depth=2, min_samples_split=20, random_state=123) # SALVA O REGRESSOR

rgr.fit(x_combustivel_train, y_combustivel_train) # TREINA O REGRESSOR



"""Calcula a acurácia para o treino e para o teste relacionados à árvore de decisão"""

from sklearn.model_selection import cross_val_score # IMPORTA A ACURACIA DO MODELO

acuracia_rgr_train = cross_val_score(rgr, x_combustivel_train, y_combustivel_train, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA PARA O TREINO
print(acuracia_rgr_train)

print('\n')

acuracia_rgr_test = cross_val_score(rgr, x_combustivel_test, y_combustivel_test, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA PARA O TESTE
print(acuracia_rgr_test)

"""Plota em forma de árvore"""

import matplotlib as mpl # IMPORTA O MATPLOTLIB
mpl.rcParams['figure.dpi'] = 100 # CONFIGURA O TAMANHO DA IMAGEM
import matplotlib.pyplot as plt # IMPORTA O PYPLOT
from sklearn.tree import plot_tree # IMPORTE PARA PLOTAR AS ÁRVORES
from sklearn.metrics import ConfusionMatrixDisplay # IMPORTE DA MATRIZ DE CONFUSÃO
from sklearn.metrics import classification_report # IMPORTA O RELATÓRIO

plt.figure() # PLOTA EM FORMA DE FIGURA
plot_tree(rgr, filled=False) # PLOTA EM FORMA DE ÁRVORE
plt.show() # PLOTA A ARVORE DE DECISÃO

"""Salva e treina o regressor, e calcula a acurácia para teste e para treino relacionadas ao modelo de floresta aleatória."""

from sklearn.ensemble import RandomForestRegressor # IMPORTA A FLORESTA ALEATÓRIA

rgr = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=123, n_jobs=-1) # SALVA O REGRESSOR

rgr.fit(x_combustivel_train, y_combustivel_train) # TREINANDO O REGRESSOR

acuracia_rgr_train = cross_val_score(rgr, x_combustivel_train, y_combustivel_train, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA PARA O TREINO
print(acuracia_rgr_train)

print('\n')

acuracia_rgr_test = cross_val_score(rgr, x_combustivel_test, y_combustivel_test, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA PARA O TESTE
print(acuracia_rgr_test)

"""Plota em forma de árvore."""

plt.figure() # PLOTA EM FORMA DE FIGURA
plot_tree(rgr.estimators_[499], filled=True) # PLOTA EM FORMA DE ARVORE
plt.show() # PLOTA A FLORESTA ALEATORIA

"""Salva e treina o regressor, calculando a acurácia para os dados de treino e teste relacionados ao XGBoost."""

import xgboost as xgb # IMPORTA O XGBOOST

rgr = xgb.XGBRegressor(n_estimators=500, max_depth=5, random_state=123, n_jobs=-1, objective="reg:squarederror") # SALVA O REGRESSOR

rgr.fit(x_combustivel_train, y_combustivel_train) # TREINA O REGRESSOR
acuracia_rgr_train = cross_val_score(rgr, x_combustivel_train, y_combustivel_train, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA DO TREINO
print(acuracia_rgr_train)

print('\n')

acuracia_rgr_test = cross_val_score(rgr, x_combustivel_test, y_combustivel_test, scoring="neg_root_mean_squared_error").mean() # CALCULA A ACURACIA DO TESTE
print(acuracia_rgr_test)

"""Plota a árvore."""

mpl.rcParams['figure.dpi'] = 100 # CONFIGURA O TAMANHO DA IMAGEM

plt.figure() # PLOTA EM FORMA DE FIGURA
xgb.plot_tree(rgr) # PLOTA EM FORMA DE ARVORE
plt.show() # PLOTA O XGBOOST GERADO

"""Fazemos vários testes, modificando o tamanho do conjunto de treino, e mantendo o conjunto de teste. O tamanho do conjunto de teste mantém em 0.25, e o conjunto de treino varia de 0.10 a 0.75. Fazemos os testes para os três modelos e mostramos abaixo, sendo árvore de decisão, floresta aleatória e XGBoost, respectivamente."""

arvore_decisao = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/COMBUSTIVEL/Regressão/Comparação Modelo - Árvore de Decisão - Regressão.csv")

arvore_decisao

floresta_aleatoria = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/COMBUSTIVEL/Regressão/Comparação Modelo - Floresta Aleatória - Regressão.csv")

floresta_aleatoria

xgboost = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/COMBUSTIVEL/Regressão/Comparação Modelo - XGBoost - Regressão.csv")

xgboost