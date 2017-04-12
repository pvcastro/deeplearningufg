import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i]
                                                       )) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i +
                                                                         1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            print(X.shape[0], a)

            for l in range(len(self.weights)):
                hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                a.append(hidden_inputs)
            error = y[i] - a[-1][:-1]
            deltas = [error * self.activation_deriv(a[-1][:-1])]
            l = len(a) - 2

            # The last layer before the output is handled separately because of
            # the lack of bias node in output
            deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            for l in range(len(a) - 3, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1][:-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights) - 1):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta[:, :-1])
            # Handle last layer separately because it doesn't have a bias unit
            i += 1
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights)):
            temp = np.ones(a.shape[0] + 1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))
        return a

nn = NeuralNetwork([2,2,1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y, epochs=10)
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.predict(i))