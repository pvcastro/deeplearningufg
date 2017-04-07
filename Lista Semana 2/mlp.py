import numpy as np
import matplotlib.pyplot as plt
import math

debug = False

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
        log("Potencial de ativação do neurônio", self.id, ": ", net_neuronio)
        self.saida = sigmoid(net_neuronio)
        log("Saída do neurônio", self.id, ": ", self.saida)
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

    def __init__(self, neuronios_por_camada, entradas, desejados, epocas, taxa_aprendizagem, precisao, pesos=None, debug_training=False, plot=True):
        global debug
        debug = debug_training
        self.plot = plot
        if not pesos:
            pesos = []
            for i in range(len(neuronios_por_camada)):
                ## Adiciona pesos aleatórios de acordo com o número de neurônios especificados, adicionando mais um peso para o bias
                pesos_da_camada = (2 * np.random.random((neuronios_por_camada[i], neuronios_por_camada[i]+1)) - 1) * 0.25
                pesos.append(pesos_da_camada)
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
        self.definir_entradas(np.array(entradas))

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
        fig, axis = plt.subplots()
        axis.set_xlabel('Épocas')
        while (epoca < self.epocas and not variacao_erro_atingida):
            print("============================ Época " + str(epoca + 1) + " ================================")
            self.propagar_sinal()
            self.retro_propagar_sinal()
            erro_anterior = erro_atual
            erro_atual = self.calcular_erro()
            plt.plot(epoca + 1, erro_atual, marker='.', )
            print("Erro quadrático médio depois de", epoca + 1, "época(s)", erro_atual)
            variacao_erro = abs(erro_atual - erro_anterior)
            print("Delta do erro depois de", epoca + 1, "época(s)", variacao_erro)
            variacao_erro_atingida = variacao_erro <= self.precisao
            if not variacao_erro_atingida:
                epoca += 1
        if self.plot:
            plt.show()
