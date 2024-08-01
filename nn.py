import random
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) ** 2 * np.exp(-z)


class Network:
    def __init__(self, *sizes: int) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def backward(self, x, y):
        # the derivatives
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]

        cur_activation = x
        activations = [x]

        # weighted inputs at each layer
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, cur_activation) + b
            zs.append(z)
            cur_activation = sigmoid(z)
            activations.append(cur_activation)

        # dC/dz, where z is weighted inputs to last layer
        delta = 2 * (activations[-1] - y) * sigmoid_prime(zs[-1])

        # dC/db, where b is the vector of biases in the last year
        dbs[-1] = delta

        # dC/dw, where w is the weight matrix for the last layer
        dws[-1] = np.dot(delta, activations[-2].transpose())

        # propogate backwards, using the equations that relate the inputs/parameters
        # at each layer to the one after it
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            dbs[-l] = delta
            dws[-l] = np.dot(delta, activations[-l - 1].transpose())

        return dbs, dws

    def stochastic_gradient_descent(
        self, training_data, epochs, batch_size, eta, test_data
    ):
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + batch_size] for k in range(0, n, batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta)
            print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")

    def update_batch(self, batch, eta):
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            db, dw = self.backward(x, y)
            dbs = [old + delta for old, delta in zip(dbs, db)]
            dws = [old + delta for old, delta in zip(dws, dw)]

        self.biases = [b - eta / len(batch) * db for b, db in zip(self.biases, dbs)]
        self.weights = [w - eta / len(batch) * dw for w, dw in zip(self.weights, dws)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
