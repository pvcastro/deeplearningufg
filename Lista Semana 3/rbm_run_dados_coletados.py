import numpy as np
from rbm import RestrictedBoltzmannMachine

#Filmes: Cassino, Intocáveis, Poderoso Chefão, Grease, Noviça Rebelde, LaLa Land

pessoa_1 = [0,1,0,1,1,0]
pessoa_2 = [0,0,0,1,1,1]
pessoa_3 = [0,0,1,1,0,1]
pessoa_4 = [1,1,1,0,0,0]
pessoa_5 = [0,1,1,0,1,0]
pessoa_6 = [0,1,1,1,0,0]
pessoa_7 = [1,1,1,0,0,0]
pessoa_8 = [0,0,1,1,0,1]
pessoa_9 = [1,1,1,0,0,0]
entradas = np.array([pessoa_1, pessoa_2, pessoa_3, pessoa_4, pessoa_5, pessoa_6, pessoa_7, pessoa_8, pessoa_9])

mlp = RestrictedBoltzmannMachine(
    entradas=entradas, quantidade_neuronios_ocultos=2, epocas=10000,
    taxa_aprendizagem=0.1, precisao=0, debug=False, plot=True
)

mlp.treinar()

print(mlp.pesos)

#Pessoa que assistiria Grease e LaLa Land
dados_de_teste = np.array([[0,0,0,1,0,1]])
mlp.prever(amostras=dados_de_teste)

#Pessoa que assistiria Intocáveis e Poderoso Chefão
dados_de_teste = np.array([[0,1,1,0,0,0]])
mlp.prever(amostras=dados_de_teste)