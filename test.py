# project/test.py

import unittest
from FFNNImplementation import Model, NeuralNetwork, Layer, Neuron, NeuronEdge

class TestCalculations(unittest.TestCase):

    def setup_model(self):
        model = Model( [2,2,1] )

        # Input Layer
        model.network.get_layer(0).get_neuron(0).bias = 1
        model.network.get_layer(0).get_neuron(1).bias = 2

        # Hidden Layer
        model.network.get_layer(1).get_neuron(0).bias = 3
        model.network.get_layer(1).get_neuron(1).bias = 4

        # Output Layer
        model.network.get_layer(2).get_neuron(0).bias = 5

        return model

    def test_connections(self):
        model = self.setup_model()

        self.assertEqual(model.network.get_number_of_layers(), 3)

        input_neurons = []
        input_neurons.append(model.get_input_neuron(0))
        input_neurons.append(model.get_input_neuron(1))

        self.assertEqual(input_neurons[0].bias, 1)
        self.assertEqual(input_neurons[1].bias, 2)

        self.assertEqual(input_neurons[0].get_child_neuron(0).bias, 3)
        self.assertEqual(input_neurons[0].get_child_neuron(1).bias, 4)

        self.assertEqual(input_neurons[1].get_child_neuron(0).bias, 3)
        self.assertEqual(input_neurons[1].get_child_neuron(1).bias, 4)

        self.assertEqual(input_neurons[0].get_child_neuron(0).get_child_neuron(0).bias, 5)
        self.assertEqual(input_neurons[1].get_child_neuron(0).get_child_neuron(0).bias, 5)

        self.assertEqual(input_neurons[0].get_child_neuron(1).get_child_neuron(0).bias, 5)
        self.assertEqual(input_neurons[1].get_child_neuron(1).get_child_neuron(0).bias, 5)

        output_node = input_neurons[0].get_child_neuron(0).get_child_neuron(0)

        self.assertEqual(output_node.bias, 5)

        self.assertEqual(output_node.get_parent_neuron(0).bias, 3)
        self.assertEqual(output_node.get_parent_neuron(1).bias, 4)

        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(0).bias, 1)
        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(1).bias, 2)

        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(0).bias, 1)
        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(1).bias, 2)

        # model.draw()

if __name__ == '__main__':
    unittest.main()

