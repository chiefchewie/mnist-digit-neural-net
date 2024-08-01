import nn
import loader

training, validation, test = loader.load_data_wrapper()

net = nn.Network(784, 30, 10)

net.stochastic_gradient_descent(training, 30, 10, 3.0, test)
