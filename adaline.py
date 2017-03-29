
# coding: utf-8

# In[14]:

import numpy as np
import matplotlib.pyplot as plt


# In[15]:

def funcao_ativacao(potencial_ativacao_u):
    return potencial_ativacao_u >= 0


# In[16]:

def calcular_erro_medio(pesos_w, desejados_d, np_entradas, debug = False):
    qtd_entradas = len(np_entradas);
    if debug:
        print("entradas: " + str(qtd_entradas))
    erros = []
    for i in range(qtd_entradas):
        entrada = np_entradas[i]
        desejado = np_desejados[i]
        potencial_ativacao_u = np.dot(entrada, pesos_w)
        erro_quadrado = (potencial_ativacao_u - desejado)**2
        if debug:
            print("erro quadrado: " + str(erro_quadrado))
        erros.append(erro_quadrado)
    erro_medio = np.average(erros)
    if debug:
        print("erro médio: " + str(erro_medio))
    return erro_medio


# In[17]:

def plotar_fronteira(pesos, entradas, desejados):
    plt.style.use('ggplot')

    b, w1, w2 = pesos
    print("pesos:")
    print(b, w1, w2)
    x = -b / w1
    y = -b / w2
    d = y
    c = -y / x

    line_x_coords = np.array([0, x])
    print("coordenadas x:")
    print(line_x_coords)
    line_y_coords = c * line_x_coords + d
    print("coordenadas y:")
    print(line_y_coords)

    plt.plot(line_x_coords, line_y_coords)
    plt.scatter(entradas[:, 0], entradas[:, 1], c=desejados, s=75)
    plt.show()


# In[18]:

def adaline(iteracoes_maxima, erros, taxa_aprendizagem, entradas_x, pesos_w, desejados_d, precisao, plot = True, debug = False):
    ##Adiciona um coeficiente 1 para o bias na matriz de entradas
    np_entradas = np.c_[np.ones(entradas_x.shape[0]), entradas_x]
    if debug:
        print("entradas: " + str(np_entradas))
    epoca = 1
    erro_anterior = 0
    erro_medio = 0
    erroEsperado = False
    while (epoca < iteracoes_maxima and not erroEsperado):
        for i in range(len(np_entradas)):
            entrada = np_entradas[i]
            if debug:
                print("entrada: " + str(entrada))
            desejado = np_desejados[i]
            if debug:
                print("desejado: " + str(desejado))
            if debug:
                print("pesos: " + str(pesos_w))
            potencial_ativacao_u = np.dot(entrada, pesos_w)
            if debug:
                print("potencial de ativação u: " + str(potencial_ativacao_u))
            erro_e = potencial_ativacao_u - desejado
            if debug:
                print("erro: " + str(erro_e))
            mudanca_pesos = taxa_aprendizagem * np.dot(erro_e, entrada)
            if debug:
                print("mudança: " + str(mudanca_pesos))
            pesos_w += mudanca_pesos
            if debug:
                print("novos pesos: " + str(pesos_w))
        erro_anterior = erro_medio
        erro_medio = calcular_erro_medio(pesos_w, desejados_d, np_entradas, debug=debug)
        variacao_erro = abs(erro_medio - erro_anterior)
        if debug:
            print("variação do erro: " + str(variacao_erro))
        erroEsperado = variacao_erro <= precisao
        if plot:
            if ((epoca == 1 or epoca % 50 == 0) or (erroEsperado)):
            #if ((epoca < 50) or (erroEsperado)):                
                print("época " + str(epoca))
                plotar_fronteira(pesos_w, entradas_x, desejados_d)
        if (not erroEsperado):
            epoca += 1
            if debug:
                print("tem erros, avançando para época " + str(epoca))
    return pesos_w, epoca, erros


# In[22]:

##Trata bias como um peso w0
pesos_w = np.array([-.8649, .3192, .3129])
np_desejados = np.array([0, 1, 1, 1])
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
pesos, epocas, erros = adaline(iteracoes_maxima=500, erros=[], taxa_aprendizagem=0.1, entradas_x=entradas, pesos_w=pesos_w, desejados_d=np_desejados, precisao = 0, plot = False, debug = True)
print("final:\npesos %s\n%s épocas" % (str(pesos), str(epocas)))


# In[20]:

##Testa função de ativação com pesos finais
np_entradas = np.c_[ np.ones(entradas.shape[0]), entradas ]
funcao_ativacao(np.dot(np_entradas, pesos))


# In[21]:

##Trata bias como um peso w0
pesos_w = np.zeros(3)
np_desejados = np.array([0, 1, 1, 1])
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
pesos, epocas, erros = adaline(iteracoes_maxima=500, erros=[], taxa_aprendizagem=0.1, entradas_x=entradas, pesos_w=pesos_w, desejados_d=np_desejados, precisao = 0, plot = True, debug = False)
print("final:\npesos %s\n%s épocas" % (str(pesos), str(epocas)))


# In[ ]:



