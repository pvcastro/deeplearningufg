import numpy as np
import matplotlib.pyplot as plt
import math

debug = True

def sigmoid(u):
    return 1 / (1 + math.exp(-u))

def log(self, *args):
    if debug == True:
        print(self, *args)

class Neuronio(object):

    def __init__(self, indice, camada, pesos, taxa_aprendizagem, desejado=0.0):
        self.indice = indice
        self.camada = camada
        self.id = "[" + str(self.camada) + "][" + str(self.indice) + "]"
        self.pesos = pesos
        self.desejado = desejado
        self.taxa_aprendizagem = taxa_aprendizagem
        self.entradas = []
        self.saida = 0
        self.erro = 0

    def definir_entradas(self, entradas):
        ''' Adiciona um coeficiente 1 para o bias na matriz de entradas '''
        self.entradas = np.append(1, entradas)

    def propagar_sinal(self):
        net_neuronio = np.dot(self.entradas, self.pesos)
        self.saida = sigmoid(net_neuronio)
        log(self.id, ": ", self.saida)
        return self.saida

    def calcular_derivada_erro_funcao_saida(self):
        return -(self.desejado - self.saida)

    def calcular_derivada_saida_funcao_net(self):
        return self.saida * (1 - self.saida)

    def retro_propagar_sinal(self, derivada_erro_funcao_saida):
        log("--------------------------------------------------------------------------------------------")
        log("Retropropagando", self.id)
        log("Entradas do neurônio", self.entradas)
        derivada_saida_funcao_net = self.calcular_derivada_saida_funcao_net()
        log("Derivada da saída em função do net", derivada_saida_funcao_net)
        derivada_erro_funcao_pesos = derivada_erro_funcao_saida * derivada_saida_funcao_net * self.entradas
        log("Derivada do erro em função dos pesos", derivada_erro_funcao_pesos)
        variacao = self.taxa_aprendizagem * derivada_erro_funcao_pesos
        log("Variação dos pesos", variacao)
        self.pesos = self.pesos - variacao
        print("Novos pesos", self.id, self.pesos)
        log("-------------------------------Fim da Retropropagação---------------------------------------")

    def calcular_erro(self):
        self.erro = np.square(self.desejado - self.saida)
        log("Erro", self.id, self.erro)
        return self.erro

class Camada(object):

    def __init__(self, quantidade_de_neuronios):
        self.neuronios = [None] * quantidade_de_neuronios
        self.saidas = []

    def definir_entradas(self, entradas):
        for neuronio in self.neuronios:
            neuronio.definir_entradas(np.array(entradas))

    def propagar_sinal(self):
        self.saidas = []
        for neuronio in self.neuronios:
            self.saidas.append(neuronio.propagar_sinal())
        log(self.saidas)
        return self.saidas

class MultiLayerPerceptron(object):

    def __init__(self, neuronios_por_camada, pesos, taxa_aprendizagem, desejados, epocas, precisao):
        self.epocas = epocas
        self.precisao = precisao
        self.camadas = []
        for indice_camada, quantidade_de_neuronios in enumerate(neuronios_por_camada):
            camada = Camada(quantidade_de_neuronios=quantidade_de_neuronios)
            for indice in range(quantidade_de_neuronios):
                #Se for a última camada, adiciona os valores desejados
                if indice_camada == (len(neuronios_por_camada) - 1):
                    neuronio = Neuronio(indice=indice, camada=indice_camada, pesos=pesos[indice_camada][indice], taxa_aprendizagem=taxa_aprendizagem, desejado=desejados[indice])
                else:
                    neuronio = Neuronio(indice=indice, camada=indice_camada, pesos=pesos[indice_camada][indice], taxa_aprendizagem=taxa_aprendizagem)
                camada.neuronios[indice] = neuronio
            self.camadas.append(camada)

    def definir_entradas(self, entradas):
        camada_entrada = self.camadas[0]
        for neuronio in camada_entrada.neuronios:
            neuronio.definir_entradas(np.array(entradas))

    def propagar_sinal(self):
        for idx, camada in enumerate(self.camadas):
            saidas = camada.propagar_sinal()
            log(saidas)
            #Se não for a última camada, seta as saídas obtidas como entrada da próxima camada
            if idx != (len(self.camadas) - 1):
                proxima_camada = self.camadas[idx + 1]
                proxima_camada.definir_entradas(np.array(saidas))

    def retro_propagar_sinal(self):
        camada_saida = self.camadas[-1]
        log("--------------------------------------------------------------------------------------------")
        derivadas_erro_funcao_net_por_pesos = []
        log("Retropropagando sinal na camada de saída")
        for neuronio in camada_saida.neuronios:
            derivada_erro_funcao_saida = neuronio.calcular_derivada_erro_funcao_saida()
            log("Derivada do erro em função da saída", derivada_erro_funcao_saida)
            derivada_saida_funcao_net = neuronio.calcular_derivada_saida_funcao_net()
            log("Derivada da saída em função do net", derivada_saida_funcao_net)
            derivada_erro_funcao_net = derivada_erro_funcao_saida * derivada_saida_funcao_net
            log("Derivada do erro em função do net", derivada_erro_funcao_net)
            log("Pesos do neurônio", neuronio.id, neuronio.pesos)
            #Desconsidera o bias na multiplicação das derivadas pelos pesos do neurônio
            derivadas_erro_funcao_net_por_pesos.append(neuronio.pesos[1:] * derivada_erro_funcao_net)
            # faz retropropagação da saída somente depois de ter guardado os dados com os pesos atuais
            neuronio.retro_propagar_sinal(derivada_erro_funcao_saida)

        derivadas_erro_funcao_net_por_pesos = np.transpose(derivadas_erro_funcao_net_por_pesos)

        for k, camada in enumerate(self.camadas[0:-1]):
            log("--------------------------------------------------------------------------------------------")
            log("Retropropagando camada intermediária", k)
            for n, neuronio in enumerate(camada.neuronios):
                derivadas_parciais = derivadas_erro_funcao_net_por_pesos[n]
                log("Para neurônio", neuronio.id, derivadas_parciais)
                soma_das_derivadas = np.sum(derivadas_parciais)
                log("Soma das derivadas parciais dos erros", soma_das_derivadas)
                neuronio.retro_propagar_sinal(soma_das_derivadas)

    def calcular_erro(self):
        erros = []
        for neuronio_saida in self.camadas[-1].neuronios:
            erros.append(neuronio_saida.calcular_erro())
        return np.average(erros)

    def treinar(self):
        erro_atual = 0
        erro_anterior = 0
        variacao_erro_atingida = False
        epoca = 0
        while (epoca < self.epocas and not variacao_erro_atingida):
            print("========== Época " + str(epoca + 1) + " ==========")
            self.propagar_sinal()
            self.retro_propagar_sinal()
            erro_anterior = erro_atual
            erro_atual = self.calcular_erro()
            plt.plot(epoca + 1, erro_atual, marker='.')
            print("Erro quadrático médio depois de", epoca + 1, "época(s)", erro_atual)
            variacao_erro = abs(erro_atual - erro_anterior)
            print("Delta do erro depois de", epoca + 1, "época(s)", variacao_erro)
            variacao_erro_atingida = variacao_erro <= self.precisao
            if not variacao_erro_atingida:
                epoca += 1
        plt.show()


mlp = MultiLayerPerceptron(
    neuronios_por_camada=[2,2],
    # Pesos e bias da camada hidden
    pesos=[[np.array([0.35, 0.15, 0.2]), np.array([0.35, 0.25, 0.3])],
           [np.array([0.6, 0.4, 0.45]), np.array([0.6, 0.5, 0.55])]],
    # taxa de aprendizagem e saidas desejáveis
    taxa_aprendizagem=0.5, desejados=np.array([0.01, 0.99]),
    epocas=1000, precisao=1e-7
)
mlp.definir_entradas(np.array([0.05, 0.1]))

debug = False

mlp.treinar()
