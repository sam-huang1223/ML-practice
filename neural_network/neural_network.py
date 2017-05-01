import numpy as np
from bokeh.plotting import figure, show, output_file, ColumnDataSource
import matplotlib.pyplot as plt

def vectorize(Y):
    vector = np.zeros(10, dtype=int)
    vector[Y] = 1
    return vector

with open('./train_x.txt') as x:
    train_x = np.array([element.split(' ') for element in x.readlines()[0].strip('[]').split('],[')], dtype=float)

with open('./train_Y.txt') as Y:
    train_Y = np.array([vectorize(int(element)) for element in Y.readlines()[0].split(',')], dtype=float)


class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights
        self.activities = {}
        self.activations = {}
        self.deltas = {}
        self.cost_record = []
        self.batch_size = 2
        self.learning_rate = 1
        # mini-batch stochastic gradient descent with a batch size of 2

    def forward_pass(self, x, layer=0):
        if layer == 0:
            self.activities[0] = x.reshape(1, len(self.weights[layer][0]))
        if layer < len(self.weights):
            activation = np.array([np.dot(neuron_weights, x) for neuron_weights in self.weights[layer]])
            self.activations[layer + 1] = activation.reshape(1, len(self.weights[layer]))
            activity = self.sigmoid(activation)
            self.activities[layer + 1] = np.array([activity]).reshape(1, len(self.weights[layer]))
            return self.forward_pass(activity, layer + 1)
        else:
            return x
#           return self.softmax(x)

#    def softmax(self, x):
#        denominator = np.sum(np.exp(x))
#        return np.exp(x)/denominator

    def train(self, train_x, train_Y, epochs):
        assert len(train_x) == len(train_Y), 'Training features and training labels have different lengths'
        assert len(train_x) % self.batch_size == 0, 'Batch size inappropriate for training set'

        before_cost = np.sum([self.cost_function(x, Y) for x, Y in zip(train_x, train_Y)])
        self.cost_record.append(before_cost)
        for epoch_num in range(epochs):
            c = list(zip(train_x, train_Y))
            np.random.shuffle(c)
            temp_train_x, temp_train_Y = zip(*c)

            for example_num in range(0, len(temp_train_x), self.batch_size):
                derivatives = np.sum([self.calculate_derivatives(temp_train_x[i], temp_train_Y[i], len(self.weights), [])
                                      for i in range(example_num, example_num + self.batch_size)], axis=0)
                self.weights = self.weights - np.multiply(self.learning_rate, derivatives)

            after_cost = np.sum([self.cost_function(x, Y) for x, Y in zip(temp_train_x, temp_train_Y)])
            self.cost_record.append(after_cost)

            if epoch_num % 100 == 0:
                print(epoch_num, '/', epochs, 'Epochs Completed')

        # Depending on how we use our data, it might not matter if or cost function is convex or not.
        # If we use our examples one at a time (stochastic GD) instead of all at once (batch GD),
        # sometimes it won't matter if our cost function is convex, we will still find a good solution.

    def calculate_derivatives(self, x, Y, layer, derivatives):
        if layer == 0:
            derivatives.reverse()
            return derivatives

        if layer == len(self.weights):
            delta = np.multiply(-(Y - self.forward_pass(x)), self.sigmoid_derivative(self.activations[layer]))
        else:
            delta = np.dot(self.deltas[layer + 1], self.weights[layer]) * self.sigmoid_derivative(self.activations[layer])
        derivative = np.dot(self.activities[layer - 1].T, delta)
        self.deltas[layer] = delta
        derivatives.append(derivative.T)
        return self.calculate_derivatives(x, Y, layer-1, derivatives)

    def cost_function(self, x, Y):
        # cost is a function of 1) examples (x) and 2) weights (forward_pass function)
        return 0.5 * ((Y - self.forward_pass(x)) ** 2)
        #One reason we chose our cost function to be the sum of squared error is to exploit the convex nature
        #of quadratic equations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

if __name__ == '__main__':
    output_node_shown = []
    ## can show only connections to output_node_shown = 7

    with open('./in.txt') as inputfile:
        test_x = np.array([element.split(' ') for element in inputfile.readlines()[0].strip('[]').split('],[')], dtype=float)

    # 7 input -> 5 hidden -> 10 output (vectorized output)
    layers = [7, 5, 10]
    ## works with arbitrary number of layers
    weights = np.array([np.random.randn(layers[i+1], layers[i]) for i in range(len(layers)-1)])

    NN = NeuralNetwork(weights)

    kwargs = dict(train_x=train_x, train_Y=train_Y, epochs=1000)
    NN.train(**kwargs)

    print('Model training completed')

    xs = np.arange(kwargs['epochs']+1)
    plt.plot(xs, NN.cost_record)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()

    with open('out.txt', 'w') as outputfile:
        for x in test_x:
            model_output = NN.forward_pass(x)
            outputfile.write(str(np.argmax(model_output)) + '\n')

    print('Output written to file')

    ### Visualization of neural network structure and weights ###

    max_layers = max(layers)
    midway_layers = max_layers / 2 if max_layers % 2 == 0 else (max_layers + 1) / 2

    plot = figure(plot_height=600, plot_width=600, x_range=(0, len(layers) + 1),
                  y_range=(0, max_layers), x_axis_type=None, y_axis_type=None,
                  toolbar_location='right', title='Final Neural Network Weights Visualization')

    scatter_xs = []
    scatter_ys = []

    for layer_num, layer in enumerate(layers):
        for neuron_num in np.arange(midway_layers - (layer-1)/2, midway_layers + (layer-1)/2 + 1, 1):
            scatter_xs.append(layer_num + 1)
            scatter_ys.append(neuron_num)

    scatter_source = ColumnDataSource({'x': scatter_xs, 'y': scatter_ys})
    plot.scatter(x='x', y='y', size = 15, color='navy', source=scatter_source)

    line_xs = []
    line_ys = []
    alphas = []
    colors = []

    layered_scatter_ys = [scatter_ys[:layers[0]]]
    for i in range(1, len(layers)):
        scatter_ys = scatter_ys[layers[i-1]:]
        layered_scatter_ys.append(scatter_ys[:layers[i]])

    for layer_num in range(len(layers) - 1):
        max_weight = np.max(NN.weights[layer_num])
        min_weight = np.min(NN.weights[layer_num])
        for from_neuron in range(layers[layer_num]):
            for to_neuron in range(layers[layer_num + 1]):
                if output_node_shown != []:
                    if layer_num == len(layers)-2 and to_neuron != output_node_shown:
                        continue
                line_xs.append([layer_num + 1, layer_num + 2])
                line_ys.append([layered_scatter_ys[layer_num][-(from_neuron + 1)],
                                layered_scatter_ys[layer_num + 1][-(to_neuron + 1)]])
                weight = NN.weights[layer_num][to_neuron][from_neuron]
                if weight < 0:
                    colors.append('cyan')
                    alphas.append(weight/min_weight)
                else:
                    colors.append('crimson')
                    alphas.append(weight/max_weight)

    connections_source = ColumnDataSource({'xs': line_xs, 'ys': line_ys, 'color': colors, 'alpha': alphas})
    plot.multi_line('xs', 'ys', color='color', alpha='alpha', source=connections_source, line_width=6)

    output_file('neural_net_visualization.html')
    #show(plot)
