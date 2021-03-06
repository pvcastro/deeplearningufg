import numpy as np
from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    numero_de_entradas=2, neuronios_por_camada=[2, 2],
    pesos=[[np.array([0.35, 0.15, 0.2]), np.array([0.35, 0.25, 0.3])],
           [np.array([0.6, 0.4, 0.45]), np.array([0.6, 0.5, 0.55])]],
    taxa_aprendizagem=0.5, epocas=2, precisao=1e-7, debug_training=True, plot=False
)

# mlp = MultiLayerPerceptron(
#     neuronios_por_camada=[2,2],
#     pesos=[[np.zeros(3), np.zeros(3)],
#            [np.zeros(3), np.zeros(3)]],
#     taxa_aprendizagem=0.5, desejados=np.array([0.01, 0.99]),
#     epocas=10, precisao=1e-7,
#     debug_training=True, plot=False
# )
# mlp = MultiLayerPerceptron(
#     neuronios_por_camada=[2,2],
#     taxa_aprendizagem=0.5, desejados=np.array([0.01, 0.99]),
#     epocas=1, precisao=1e-7
# )

mlp.treinar(matriz_entradas=np.array([[0.05, 0.1]]), valores_desejados=np.array([[0.01, 0.99]]))