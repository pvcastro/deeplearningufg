# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:54:45 2017

@author: www.deeplearningbrasil.com.br
"""

# Rede Neural convolucional simples para o problema de reconhecimento de dígitos (MNIST)
import tensorflow as tf
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import roc_curve, auc

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Verifique o site https://www.tensorflow.org/get_started/mnist/beginners para
# mais informações sobre o conjunto de dados

# Parâmetros de aprendizagem
taxa_aprendizado = 0.001
quantidade_maxima_epocas = 15
batch_size = 200

#with tf.device('/cpu:0'):

# entrada dos place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# imagem 28x28x1 (preto e branca)
X_img = tf.reshape(X, [-1, 28, 28, 1])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

# Classificador - Camada Fully Connected entrada 7x7x64 e 1024 neurônios
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.relu(tf.matmul(L2_flat, W3))

keep_prob = tf.placeholder(tf.float32)
L3_drop = tf.nn.dropout(L3, keep_prob)

# Classificador - Camada Fully Connected entrada 1024 -> 10 saídas
W4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.01))
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_drop, W4) + b

# Define a função de custo e o método de otimização
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=taxa_aprendizado).minimize(cost)

# inicializa
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# treina a rede
print('Rede inicialiada. Treinamento inicializado. Tome um cafe...')
for epoca in range(quantidade_maxima_epocas):
    custo_medio = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.2}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        custo_medio += c / total_batch

    print('Epoca:', '%04d' % (epoca + 1), 'perda =', '{:.9f}'.format(custo_medio))

print('Treinamento finalizado!')

# Teste o modelo e verifica a taxa de acerto
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Taxa de acerto:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

# Obtém uma nova imagem e testa o modelo
r = random.randint(0, mnist.test.num_examples - 1)
print("Classe real: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Predição: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0}))

# for i in range(mnist.test.num_examples):
#     y_p = tf.argmax(logits, 1)
#     x_val = mnist.test.images[:,i]
#     print(x_val.shape)
#     y_val = mnist.test.labels[:,i]
#     print(y_val.shape)
#     val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X: x_val, Y: y_val, keep_prob: 1.0})
#     print(y_pred)
#     fpr, tpr, tresholds = roc_curve(y_val, y_pred)
#     print("class %s:auc=%s" % (i, auc(fpr, tpr)))

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(10):
#     fpr[i], tpr[i], thresh = roc_curve(valid.y[:, i], prediction[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     print("class %s:auc=%s" % (i, roc_auc[i]))