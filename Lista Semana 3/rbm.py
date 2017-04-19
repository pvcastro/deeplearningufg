import numpy as np
import matplotlib.pyplot as plt

class RestrictedBoltzmannMachine(object):

    def __init__(self, entradas, quantidade_neuronios_ocultos, epocas, taxa_aprendizagem, precisao,
                 estados_neuronios=None, pesos=None, debug=False, plot=True):
        self.debug = debug
        self.entradas = np.insert(entradas, 0, 1, axis=1)
        self.plot = plot
        self.epocas = epocas
        self.taxa_aprendizagem = taxa_aprendizagem
        self.precisao = precisao
        self.estados_neuronios = estados_neuronios
        self.quantidade_neuronios_ocultos = quantidade_neuronios_ocultos
        if pesos is None:
            pesos = (2 * np.random.random((self.entradas.shape[1], quantidade_neuronios_ocultos + 1)) - 1) * 0.25
        self.pesos = pesos

    def log(self, *args):
        if self.debug:
            print(*args)

    def sigmoid(self, energia_ativacao):
        return 1 / (1 + np.exp(-energia_ativacao))

    def propagar_sinal(self):
        energia_ativacao_Uv = np.dot(self.entradas, self.pesos)
        self.log("Energia de ativação da propagação", "\n", energia_ativacao_Uv)
        ativacao_gUv = self.sigmoid(energia_ativacao_Uv)
        self.log("Ativação da propagação", "\n", ativacao_gUv)
        if self.estados_neuronios is None:
            numeros_aleatorios = np.random.random((ativacao_gUv.shape[0], ativacao_gUv.shape[1]))
            estados_neuronios = ativacao_gUv >= numeros_aleatorios
        else:
            estados_neuronios = self.estados_neuronios
            self.estados_neuronios = None
        self.log("Estados dos neurônios", "\n", estados_neuronios)
        return ativacao_gUv, estados_neuronios

    def reconstruir_entrada(self, estados_neuronios):
        energia_ativacao_Uh = np.dot(estados_neuronios, self.pesos.T)
        self.log("Energia de ativação da reconstrução", "\n", energia_ativacao_Uh)
        entradas_reconstruidas = self.sigmoid(energia_ativacao_Uh)
        entradas_reconstruidas[:, 0] = 1.0
        self.log("Entradas reconstruídas", "\n", entradas_reconstruidas)
        energia_ativacao_estimada_Uv = np.dot(entradas_reconstruidas, self.pesos)
        self.log("Energia de ativação estimada da reconstrução", "\n", energia_ativacao_estimada_Uv)
        ativacao_estimada_gUv = self.sigmoid(energia_ativacao_estimada_Uv)
        self.log("Ativação estimada da reconstrução", "\n", ativacao_estimada_gUv)
        return entradas_reconstruidas, ativacao_estimada_gUv

    def calcular_associacao_positiva(self, ativacao_gUv):
        associacao_positiva = np.dot(self.entradas.T, ativacao_gUv)
        self.log("Associação positiva", "\n", associacao_positiva)
        return associacao_positiva

    def calcular_associacao_negativa(self, entradas_reconstruidas, ativacao_estimada_gUv):
        associacao_negativa = np.dot(entradas_reconstruidas.T, ativacao_estimada_gUv)
        self.log("Associação negativa", "\n", associacao_negativa)
        return associacao_negativa

    def atualizar_pesos(self, associacao_positiva, associacao_negativa):
        self.pesos = self.pesos + self.taxa_aprendizagem * ((associacao_positiva - associacao_negativa) / self.entradas.shape[0])
        self.log("Pesos atualizados", "\n", self.pesos)

    def calcular_erro(self, entradas_reconstruidas):
        erro_epoca = np.sum(np.square(self.entradas - entradas_reconstruidas))
        self.log("Erro da época", "\n", erro_epoca)
        return erro_epoca

    def treinar(self):
        erro_atual = 0
        erro_anterior = 0
        variacao_erro_atingida = False
        epoca = 0
        fig, axis = plt.subplots()
        axis.set_xlabel('Épocas')
        while (epoca < self.epocas and not variacao_erro_atingida):
            print("============================ Época " + str(epoca + 1) + " ================================")
            ativacao_gUv, estados_neuronios = self.propagar_sinal()
            entradas_reconstruidas, ativacao_estimada_gUv = self.reconstruir_entrada(
                estados_neuronios=estados_neuronios)
            associacao_positiva = self.calcular_associacao_positiva(ativacao_gUv=ativacao_gUv)
            associacao_negativa = self.calcular_associacao_negativa(entradas_reconstruidas=entradas_reconstruidas, ativacao_estimada_gUv=ativacao_estimada_gUv)
            self.atualizar_pesos(associacao_positiva=associacao_positiva, associacao_negativa=associacao_negativa)
            erro_anterior = erro_atual
            erro_atual = self.calcular_erro(entradas_reconstruidas=entradas_reconstruidas)
            plt.plot(epoca + 1, erro_atual, marker='.', )
            print("Erro depois de", epoca + 1, "época(s)", erro_atual)
            variacao_erro = abs(erro_atual - erro_anterior)
            print("Delta do erro depois de", epoca + 1, "época(s)", variacao_erro)
            variacao_erro_atingida = variacao_erro <= self.precisao
            if not variacao_erro_atingida:
                epoca += 1
        if self.plot:
            plt.show()

    def prever(self, amostras):
        quantidade_amostras = amostras.shape[0]
        # Adiciona bias nas amostras
        amostras = np.insert(amostras, 0, 1, axis=1)
        #estados_neuronios_ocultos = np.ones((quantidade_amostras, self.quantidade_neuronios_ocultos + 1))
        energia_ativacao = np.dot(amostras, self.pesos)
        ativacao = self.sigmoid(energia_ativacao)
        estados = ativacao >= np.random.rand(quantidade_amostras, self.quantidade_neuronios_ocultos + 1)
        # Despreza os bias
        estados = estados[:, 1:]
        return estados
