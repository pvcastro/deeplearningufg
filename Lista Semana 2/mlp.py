import numpy as np
import matplotlib.pyplot as plot
import math


def sigmoid(u):
    return 1 / (1 + math.exp(-u))


class Neuronio(object):

    def __init__(self, indice, camada, pesos, taxa_aprendizagem, desejado=0.0):
        self.indice = indice
        self.camada = camada
        self.pesos = pesos
        self.desejado = desejado
        self.taxa_aprendizagem = taxa_aprendizagem
        self.entradas = []
        self.saida = 0
        self.erro = 0

    def set_entradas(self, entradas):
        ##Adiciona um coeficiente 1 para o bias na matriz de entradas
        self.entradas = np.append(1, entradas)

    def propagate(self):
        """ Propaga as entradas pelo neuronio (somatório ponderado -> sigmoide) """
        summation = np.sum(self.entradas*self.pesos)
        self.saida = sigmoid(summation)
        print("[", self.camada, "]", "[", self.indice, "]", ": ", self.saida)
        return self.saida

    def calculate_e_out(self):
        """ Calcula a derivada do erro total pela saida do neuronio da camada saida """
        return -(self.desejado - self.saida)

    def calculate_e_out_net(self, e_out):
        """ Calcula a derivada da saida do neuronio pelo somatório """
        out_net = self.saida*(1-self.saida)
        return e_out*out_net

    def retro_propagate(self, e_out):
        """ Recebe e_out (derivada do erro total pela saida do neuronio em questão e recalcula os pesos """
        e_out_net = self.calculate_e_out_net(e_out)
        self.pesos = self.pesos-(self.entradas*e_out_net*self.taxa_aprendizagem)
        print("New Weights", self.name, self.pesos)

    def calculate_erro(self):
        """ Calcula o erro quadrático e retorna """
        self.erro = self.desejado-self.saida
        self.erro = (self.erro*self.erro)/2
        print("Error", self.name, self.erro)
        return self.erro

class Camada(object):
    """ Classe responsável por representar uma rede neural Multilayer Perceptron com duas camadas """

    def __init__(self, quantidade_de_neuronios):
        self.neuronios = [None] * quantidade_de_neuronios
        self.saidas = []

    def set_entradas(self, entradas):
        ''' Percorre os neurônios da camada de entrada para definir as entradas '''
        for neuronio in self.neuronios:
            neuronio.set_entradas(np.array(entradas))

    def propagate(self):
        self.saidas = []
        for neuronio in self.neuronios:
            self.saidas.append(neuronio.propagate())
        print(self.saidas)
        return self.saidas

class MultilayerPerceptron(object):
    """ Classe responsável por representar uma rede neural Multilayer Perceptron com duas camadas """

    def __init__(self, neuronios_por_camada, pesos, taxa_aprendizagem, desejados):
        self.camadas = []
        for indice_camada, quantidade_de_neuronios in enumerate(neuronios_por_camada):
            camada = Camada(quantidade_de_neuronios=quantidade_de_neuronios)
            for indice in range(quantidade_de_neuronios):
                """ Se for a última camada, adiciona os valores desejados """
                if indice_camada == (len(neuronios_por_camada) - 1):
                    neuronio = Neuronio(indice=indice, camada=indice_camada, pesos=pesos[indice_camada][indice], taxa_aprendizagem=taxa_aprendizagem, desejado=desejados[indice])
                else:
                    neuronio = Neuronio(indice=indice, camada=indice_camada, pesos=pesos[indice_camada][indice], taxa_aprendizagem=taxa_aprendizagem)
                camada.neuronios[indice] = neuronio
            self.camadas.append(camada)

    def set_entradas(self, entradas):
        ''' Percorre os neurônios da camada de entrada para definir as entradas '''
        camada_entrada = self.camadas[0]
        for neuronio in camada_entrada.neuronios:
            neuronio.set_entradas(np.array(entradas))

    def propagate(self):
        for idx, camada in enumerate(self.camadas):
            saidas = camada.propagate()
            for neuronio in camada.neuronios:
                saidas.append(neuronio.propagate())
            print(saidas)
            ''' Se não for a última camada, seta as saídas obtidas como entrada da próxima camada '''
            if idx != (len(self.camadas) - 1):
                proxima_camada = self.camadas[idx + 1]
                proxima_camada.set_entradas(np.array(saidas))
                proxima_camada.propagate()
                #for neuronio in proxima_camada:
                #    neuronio.set_entradas(np.array(saidas))

    def retro_propagate(self):
        # e_out_h é a derivada do Erro no neuronio Oi em ghj (saida do neuronio j da camada hidden)
        e_out_h = []
        for o in self.saida_layer:
            e_out = o.calculate_e_out()
            e_out_net = o.calculate_e_out_net(e_out)
            e_out_h.append(o.pesos * e_out_net)
            # retro propaga na camada de saida
            o.retro_propagate(e_out)

        # faz a transposta visto que para cada neuronio da camada de entrada é necessário o somatorio das derivadas de
        # acordo com erros de cada neuronio de saida
        e_out_h = np.transpose(e_out_h)

        for k, h in enumerate(self.hidden_layer):
            # faz o somatório das derivadas de todos os erros em ghj e propaga na camada hidden
            h.retro_propagate(np.sum(e_out_h[k]))

    def calculate_erro(self):
        """ Calcula o erro somando os erros quadráticos de cada neuronio da camada de saida """
        erro = 0
        for o in self.saida_layer:
            erro += o.calculate_erro()
        return erro


nna = MultilayerPerceptron(
    neuronios_por_camada=[2,2],
    # Pesos e bias da camada hidden
    pesos=[[np.array([0.35, 0.15, 0.2]), np.array([0.35, 0.25, 0.3])],
           [np.array([0.6, 0.4, 0.45]), np.array([0.6, 0.5, 0.55])]],
    # taxa de aprendizagem e saidas desejáveis
    taxa_aprendizagem=0.5, desejados=np.array([0.01, 0.99])
)
nna.set_entradas(np.array([0.05, 0.1]))

for i in range(2):
    print("========== Epoca "+str(i+1)+" ==========")
    nna.propagate()
    nna.retro_propagate()
    plot.plot(i+1, nna.calculate_erro(), marker='o')
    print("Total Error:", nna.calculate_erro())

plot.show()