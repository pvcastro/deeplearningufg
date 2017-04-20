import numpy as np
from rbm import RestrictedBoltzmannMachine

#Filmes: Harry Potter, Avatar, LOTR 3, Gladiador, Titanic, Glitter

aluno_1 = [0,1,1,1,0,0]
aluno_2 = [0,1,1,1,0,0]
aluno_3 = [1,1,1,0,0,0]
aluno_4 = [0,1,1,0,0,1]
aluno_5 = [0,1,1,1,0,0]
aluno_6 = [1,0,0,0,1,1]
entradas = np.array([aluno_1, aluno_2, aluno_3, aluno_4, aluno_5, aluno_6])

pesos = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.1, 0.2],
         [0.0, 0.3, 0.4],
         [0.0, 0.5, 0.6],
         [0.0, 0.7, 0.8],
         [0.0, 0.9, 1.0],
         [0.0, 0.1, 0.2]])

probabilidade_associacao_neuronios = np.array([[False, False, True],
                                               [False, True, True],
                                               [False, True, True],
                                               [True, True, True],
                                               [True, True, True],
                                               [True, True, True]])

mlp = RestrictedBoltzmannMachine(
    entradas=entradas, quantidade_neuronios_ocultos=2, epocas=5000,
    taxa_aprendizagem=0.1, precisao=0, pesos=pesos, debug=False, plot=False
)

mlp.treinar()

print(mlp.pesos)

#Pessoa que assistiria somente Gladiador e Titanic
dados_de_teste = np.array([[1,0,1,1,0,0]])
mlp.prever(amostras=dados_de_teste)
mlp.prever(amostras=entradas)