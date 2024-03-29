# James Folk
# DATA 527 – Predictive Modeling
# Assignment 3
# DEADLINE: March 28, 2024
# Spring 2024

from matplotlib import pyplot
from math import cos, sin, atan, exp
import os

## View

class NeuronView():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        font = {
            'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 8,
        }
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        pyplot.text(self.x, self.y,  self.get_neuron_text(), horizontalalignment='center', verticalalignment='center', fontdict=font)

    def get_neuron_text(self):
        return "0.0"#"{}\n{}".format(self.x, self.y)

class LayerView():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.scale = 1
        self.vertical_distance_between_layers = 6 * self.scale
        self.horizontal_distance_between_neurons = 2 * self.scale
        self.neuron_radius = 0.5 * self.scale
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __create_neuron(self):
        return NeuronView(self.x, self.y)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        self.x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = self.__create_neuron()
            neurons.append(neuron)
            self.x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetworkView():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def __create_layer(self, number_of_neurons):
        return LayerView(self, number_of_neurons, self.number_of_neurons_in_widest_layer)

    def add_layer(self, number_of_neurons ):
        layer = self.__create_layer(number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()


## Model

class NeuronEdge():
    def __init__(self, neuron, weight = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)):
        self.neuron = neuron
        self.weight = weight

    def calc():
        self.neuron.bias * self.weight

class Neuron(NeuronView):
    def __init__(self, x, y):
        super().__init__(x, y)

        # edges coming from previous layer
        self.parents = []

        # edges going to the next layer
        self.children = []

        self.bias = 0

    def get_number_of_children(self):
        return len(self.children)

    def get_child_neuron(self, index):
        if index >= 0 and index < len(self.children):
            return self.children[index].neuron
        return None

    def get_child_weight(self, index):
        if index >= 0 and index < len(self.children):
            return self.children[index].weight
        return None

    def get_number_of_parents(self):
        return len(self.parents)

    def get_parent_neuron(self, index):
        if index >= 0 and index < len(self.parents):
            return self.parents[index].neuron
        return None

    def get_parent_weight(self, index):
        if index >= 0 and index < len(self.parents):
            return self.parents[index].weight
        return None

    def get_neuron_text(self):
        return "{}".format(self.bias)

    def calculate_value(self):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        input_array = []
        for in_edge in self.parents:
            input_array.append(in_edge.calc())
        return sigmoid(sum(input_array))

class Layer(LayerView):
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        super().__init__(network, number_of_neurons, number_of_neurons_in_widest_layer)
        self.__init_layer()

    def __init_layer(self):
        for neuron in self.neurons:
            if None != self.previous_layer:
                # init children of previous layer neurons
                for prev_neuron in self.previous_layer.neurons:
                    prev_neuron.children.append(NeuronEdge(neuron))
                # init parents of current layer neurons
                for prev_neuron in self.previous_layer.neurons:
                    neuron.parents.append(NeuronEdge(prev_neuron))

    def _LayerView__create_neuron(self):
        return Neuron(self.x, self.y)

    def get_number_of_neurons(self):
        return len(self.neurons)

    def get_neuron(self, index):
        if index >= 0 and index < len(self.neurons):
            return self.neurons[index]
        return None



class NeuralNetwork(NeuralNetworkView):
    def __init__(self, number_of_neurons_in_widest_layer):
        super().__init__(number_of_neurons_in_widest_layer)

    def _NeuralNetworkView__create_layer(self, number_of_neurons):
        return Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)

    def get_number_of_layers(self):
        return len(self.layers)

    def get_layer(self, index):
        if index >= 0 and index < len(self.layers):
            return self.layers[index]
        return None


class Model():
    def __init__( self, neural_network ):
        self.neural_network = neural_network
        widest_layer = max( self.neural_network )
        self.network = NeuralNetwork( widest_layer )
        for l in self.neural_network:
            self.network.add_layer(l)

    def draw( self ):
        self.network.draw()

    def get_input_neuron(self, index):
        if self.network.get_number_of_layers() > 0 and index >= 0 and index < self.network.get_layer(0).get_number_of_neurons():
            return self.network.get_layer(0).get_neuron(index)
        return None

def main():
    model = Model( [2,2,1] )
    model.draw()

if __name__=="__main__":
    main()

