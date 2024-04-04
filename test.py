# project/test.py

import unittest
from FFNNImplementation import Model, NeuralNetwork, Layer, Neuron, NeuronEdge, calculate_r_squared
import matplotlib.pyplot as plt;
import random
import json

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


        # w1
        model.network.get_layer(0).get_neuron(0).weights[0] = 1
        # w2
        model.network.get_layer(0).get_neuron(0).weights[1] = 2
        # w3
        model.network.get_layer(0).get_neuron(1).weights[0] = 3
        # w4
        model.network.get_layer(0).get_neuron(1).weights[1] = 4
        # w5
        model.network.get_layer(1).get_neuron(0).weights[0] = 5
        # w6
        model.network.get_layer(1).get_neuron(1).weights[0] = 6

        return model

    def setup_model_clear(self):
        model = Model( [2,2,1] )

        # Input Layer
        model.network.get_layer(0).get_neuron(0).bias = 0
        model.network.get_layer(0).get_neuron(1).bias = 0

        # Hidden Layer
        model.network.get_layer(1).get_neuron(0).bias = 3
        model.network.get_layer(1).get_neuron(1).bias = 4

        # Output Layer
        model.network.get_layer(2).get_neuron(0).bias = 5


        # w1
        model.network.get_layer(0).get_neuron(0).weights[0] = 1
        # w2
        model.network.get_layer(0).get_neuron(0).weights[1] = 2
        # w3
        model.network.get_layer(0).get_neuron(1).weights[0] = 3
        # w4
        model.network.get_layer(0).get_neuron(1).weights[1] = 4
        # w5
        model.network.get_layer(1).get_neuron(0).weights[0] = 5
        # w6
        model.network.get_layer(1).get_neuron(1).weights[0] = 6

        return model

    def test_structure(self):
        model = self.setup_model()

        self.assertEqual(model.network.get_number_of_layers(), 3)

        input_neurons = []
        input_neurons.append(model.get_input_neuron(0))
        input_neurons.append(model.get_input_neuron(1))
        output_node = input_neurons[0].get_child_neuron(0).get_child_neuron(0)

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

        self.assertEqual(output_node.bias, 5)

        self.assertEqual(output_node.get_parent_neuron(0).bias, 3)
        self.assertEqual(output_node.get_parent_neuron(1).bias, 4)

        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(0).bias, 1)
        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(1).bias, 2)

        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(0).bias, 1)
        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(1).bias, 2)

        self.assertEqual(input_neurons[0].weights[0], 1)
        self.assertEqual(input_neurons[0].weights[1], 2)

        self.assertEqual(input_neurons[1].weights[0], 3)
        self.assertEqual(input_neurons[1].weights[1], 4)

        self.assertEqual(input_neurons[0].get_child_neuron(0).weights[0], 5)
        self.assertEqual(input_neurons[0].get_child_neuron(1).weights[0], 6)

        self.assertEqual(output_node.get_parent_neuron(0).weights[0], 5)
        self.assertEqual(output_node.get_parent_neuron(1).weights[0], 6)

        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(0).weights[0], 1)
        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(0).weights[1], 2)
        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(1).weights[0], 3)
        self.assertEqual(output_node.get_parent_neuron(0).get_parent_neuron(1).weights[1], 4)

        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(0).weights[0], 1)
        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(0).weights[1], 2)
        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(1).weights[0], 3)
        self.assertEqual(output_node.get_parent_neuron(1).get_parent_neuron(1).weights[1], 4)

    def test_video_lesson_function(self):
        inputs = [5, 3]
        outputs = [12]

        LR = 0.00001

        model = Model( [2,2,1] )


        for i in range(0, 100000):
            model.iterate_training(inputs, outputs, LR)
        # model.draw()

    def test_batch_gradient_descent(self):

        independentVariablesArray = [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1]
                ]
        dependantVariables = [ 0, 1, 1, 0 ]
        errorArray = []

        LR = 0.001
        N = 10000

        # Batch Gradient Descent

        model = Model( [2,2,1] )

        for n in range(N):
            error = model.iterate(independentVariablesArray, dependantVariables, LR)
            errorArray.append(error)

        predictedArray = []
        for independentVariable in independentVariablesArray:
            predictedArray.append(model.feed_forward(independentVariable))
        rValue = calculate_r_squared(dependantVariables, predictedArray)

        plt.plot(errorArray)
        plt.savefig('data/BatchGradientDescent_MSE.pdf', dpi=150)
        plt.show()

        with open("data/BatchGradientDescentModelParameters.json", 'w') as file:
            if [] == errorArray:
                dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray, "r value":rValue, "neural network": json.loads(str(model))}
            else:
                dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray[-1], "r value":rValue, "neural network": json.loads(str(model))}

            json.dump(dictionary, file, indent=2)

    def test_stochastic(self):

        independentVariablesArray = [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1]
                ]
        dependantVariables = [ 0, 1, 1, 0 ]
        errorArray = []

        LR = 0.01
        N = 10000

        # Stochastic

        model = Model( [2,2,1] )

        for n in range(N):
            random_index = random.choice(range(0, len(independentVariablesArray)))
            _independentVariablesArray = [
                    independentVariablesArray[random_index]
                    ]
            _dependantVariables = [ dependantVariables[random_index] ]

            error = model.iterate(_independentVariablesArray, _dependantVariables, LR)
            errorArray.append(error)

        predictedArray = []
        for independentVariable in independentVariablesArray:
            predictedArray.append(model.feed_forward(independentVariable))
        rValue = calculate_r_squared(dependantVariables, predictedArray)

        plt.plot(errorArray)
        plt.savefig('data/Stochastic_MSE.pdf', dpi=150)
        plt.show()

        with open("data/StochasticModelParameters.json", 'w') as file:
            if [] == errorArray:
                dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray, "r value":rValue, "neural network": json.loads(str(model))}
            else:
                dictionary={"learningRate":LR, "iterations":N, "final mse":errorArray[-1], "r value":rValue, "neural network": json.loads(str(model))}

            json.dump(dictionary, file, indent=2)

if __name__ == '__main__':
    unittest.main()

