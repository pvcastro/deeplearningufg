import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

debug = False

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def log(self, *args):
    if debug == True:
        print(self, *args)

class Neuronio(object):

    def __init__(self, indice, camada, taxa_aprendizagem):
        self.indice = indice
        self.camada = camada
        self.id = "[" + str(self.camada.indice) + "][" + str(self.indice) + "]"
        self.taxa_aprendizagem = taxa_aprendizagem
        self.entradas = []
        self.saida = 0
        self.erro = 0

    def definir_entradas(self, entradas):
        ''' Adiciona um coeficiente 1 para o bias na matriz de entradas '''
        self.entradas = np.append(1, entradas)

    def definir_desejados(self, desejados):
        self.desejados = desejados

    def propagar_sinal(self):
        indice_camada = self.camada.indice
        indice = self.indice
        net_neuronio = np.dot(self.entradas, self.camada.rede.pesos[indice_camada][indice])
        log("Potencial de ativação do neurônio", self.id, ": ", net_neuronio)
        self.saida = sigmoid(net_neuronio)
        log("Saída do neurônio", self.id, ": ", self.saida)
        return self.saida

    def calcular_derivada_erro_funcao_saida(self):
        return -(self.desejados - self.saida)

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
        indice_camada = self.camada.indice
        indice = self.indice
        self.camada.rede.pesos[indice_camada][indice] = self.camada.rede.pesos[indice_camada][indice] - variacao
        log("Novos pesos", self.id, self.camada.rede.pesos[indice_camada][indice])
        log("-------------------------------Fim da Retropropagação---------------------------------------")

    def calcular_erro(self):
        self.erro = np.square(self.desejados - self.saida)
        log("Erro", self.id, self.erro)
        return self.erro

    def get_pesos(self):
        indice_camada = self.camada.indice
        indice = self.indice
        return self.camada.rede.pesos[indice_camada][indice]

class Camada(object):

    def __init__(self, quantidade_de_neuronios, indice, rede):
        self.neuronios = [None] * quantidade_de_neuronios
        self.indice = indice
        self.rede = rede
        self.saidas = []

    def definir_entradas(self, entradas):
        for neuronio in self.neuronios:
            neuronio.definir_entradas(np.array(entradas))

    def definir_desejados(self, desejados):
        for indice, neuronio in enumerate(self.neuronios):
            if isinstance(desejados, list) or isinstance(desejados, np.ndarray):
                neuronio.definir_desejados(np.array(desejados[indice]))
            else:
                neuronio.definir_desejados(np.array(desejados))

    def propagar_sinal(self):
        self.saidas = []
        for neuronio in self.neuronios:
            self.saidas.append(neuronio.propagar_sinal())
        log(self.saidas)
        return self.saidas

class MultiLayerPerceptron(object):

    def __init__(self, numero_de_entradas, neuronios_por_camada, epocas, taxa_aprendizagem, precisao, pesos=None, debug_training=False, plot=True):
        global debug
        debug = debug_training
        self.plot = plot
        if not pesos:
            pesos = []
            # for i in range(len(neuronios_por_camada)):
            #     ## Adiciona pesos aleatórios de acordo com o número de neurônios especificados, adicionando mais um peso para o bias
            #     pesos_da_camada = (2 * np.random.random((neuronios_por_camada[i], neuronios_por_camada[i]+1)) - 1) * 0.25
            #     pesos.append(pesos_da_camada)
            for i in range(len(neuronios_por_camada)):
                ## Quantidade de neurônios é a especificada no array de neurônios por camada do índice corrente
                quantidade_de_neuronios = neuronios_por_camada[i]
                ## Quantidade de pesos é a quantidade de neurônios da camada anterior. Se for a primeira camada, usa o número de entradas.
                ## Soma-se um pelo bias
                quantidade_de_pesos = 0
                if i == 0:
                    quantidade_de_pesos = numero_de_entradas
                else:
                    quantidade_de_pesos = neuronios_por_camada[i - 1]
                pesos_da_camada = (2 * np.random.random((quantidade_de_neuronios, quantidade_de_pesos + 1)) - 1) * 0.25
                pesos.append(pesos_da_camada)
        self.pesos = pesos
        self.epocas = epocas
        self.precisao = precisao
        self.camadas = []
        for indice_camada, quantidade_de_neuronios in enumerate(neuronios_por_camada):
            camada = Camada(quantidade_de_neuronios=quantidade_de_neuronios, indice=indice_camada, rede=self)
            for indice in range(quantidade_de_neuronios):
                camada.neuronios[indice] = Neuronio(indice=indice, camada=camada, taxa_aprendizagem=taxa_aprendizagem)
            self.camadas.append(camada)

    def definir_entradas(self, entradas):
        camada_entrada = self.camadas[0]
        camada_entrada.definir_entradas(entradas)

    def definir_desejados(self, desejados):
        camada_saida = self.camadas[-1]
        camada_saida.definir_desejados(desejados)

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
            log("Pesos do neurônio", neuronio.id, neuronio.get_pesos())
            #Desconsidera o bias na multiplicação das derivadas pelos pesos do neurônio
            derivadas_erro_funcao_net_por_pesos.append(neuronio.get_pesos()[1:] * derivada_erro_funcao_net)
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

    def treinar(self, matriz_entradas, valores_desejados):
        erro_atual = 0
        erro_anterior = 0
        variacao_erro_atingida = False
        epoca = 0
        fig, axis = plt.subplots()
        axis.set_xlabel('Épocas')
        while (epoca < self.epocas and not variacao_erro_atingida):
            print("============================ Época " + str(epoca + 1) + " ================================")
            erros_por_amostra = []
            for indice, amostra in enumerate(matriz_entradas):
                self.treinar_amostra(amostra=amostra, desejados=valores_desejados[indice], erros_por_amostra=erros_por_amostra)
            erro_anterior = erro_atual
            erro_atual = np.average(erros_por_amostra)
            plt.plot(epoca + 1, erro_atual, marker='.', )
            print("Erro quadrático médio depois de", epoca + 1, "época(s)", erro_atual)
            variacao_erro = abs(erro_atual - erro_anterior)
            print("Delta do erro depois de", epoca + 1, "época(s)", variacao_erro)
            variacao_erro_atingida = variacao_erro <= self.precisao
            if not variacao_erro_atingida:
                epoca += 1
        if self.plot:
            plt.show()

    def treinar_amostra(self, amostra, desejados, erros_por_amostra):
        self.definir_entradas(amostra)
        self.definir_desejados(desejados)
        self.propagar_sinal()
        self.retro_propagar_sinal()
        erros_por_amostra.append(self.calcular_erro())

    def prever(self, amostra):
        self.definir_entradas(amostra)
        self.propagar_sinal()
        camada_de_saida = self.camadas[-1]
        return camada_de_saida.saidas

    def avaliar(self, saidas_esperadas, amostras_de_teste):
        previsoes = []
        previsoes_ajustadas = []
        for amostra_teste in amostras_de_teste:
            previsao = self.prever(amostra_teste)
            previsoes.append(previsao)
            if np.greater_equal(previsao, [0.5]):
                previsao_ajustada = 1
            else:
                previsao_ajustada = 0
            previsoes_ajustadas.append(previsao_ajustada)

        print("------------------------------------------Resultados------------------------------------------")
        print("Previsões ajustadas", previsoes_ajustadas)
        print("Previsões", previsoes)
        print("Saídas desejadas", saidas_esperadas)

        erros = 0
        tamanho = len(saidas_esperadas)
        for i in range(tamanho):
            if saidas_esperadas[i] != previsoes_ajustadas[i]:
                erros += 1
        acertos = tamanho - erros
        accuracy = float(acertos) / tamanho
        print("-----------------------------------------------------")
        print('# Registros corretamente classificados =', acertos)
        print('# Registros incorretamente classificados =', erros)
        print('# Total de registros =', tamanho)
        print("-----------------------------------------------------")
        print('Precisão = %.2f' % accuracy)
        print("")
        print("Matriz de confusão:")
        print(confusion_matrix(saidas_esperadas, previsoes_ajustadas))
        report = classification_report(y_true=saidas_esperadas, y_pred=previsoes_ajustadas,
                                       target_names=['Normal', 'Altered'])
        print("")
        print("Relatório:")
        print(report)
