import numpy as np
from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    neuronios_por_camada=[2,2],
    taxa_aprendizagem=0.5, desejados=np.array([0.01, 0.99]),
    epocas=1, precisao=1e-7
)

mlp.definir_entradas(np.array([0.05, 0.1]))

mlp.treinar(plot=False)