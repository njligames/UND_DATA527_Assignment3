# project/test.py

import unittest
from FFNNImplementation import Model, NeuralNetwork, Layer, Neuron, NeuronEdge
import matplotlib.pyplot as plt;

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

    def test_with_mse(self):
        def calculate_r_squared(y_true, y_pred):
            if len(y_true) != len(y_pred):
                raise ValueError("Length of dependant values and predicted values must be the same.")

            # Calculate the mean of the true values
            mean_y_true = sum(y_true) / len(y_true)

            # Calculate the total sum of squares (TSS) without using sum
            tss = 0
            for y in y_true:
                tss += (y - mean_y_true) ** 2

            # Calculate the residual sum of squares (RSS) without using sum
            rss = 0
            for true_val, pred_val in zip(y_true, y_pred):
                rss += (true_val - pred_val) ** 2

            # Calculate R-squared
            r_squared = 1 - (rss / tss)

            return r_squared

        independentVariablesArray = [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1]
                ]
        dependantVariables = [ 0, 1, 1, 0 ]
        errorArray = []

        LR = 0.001
        N = 1000000

        model = Model( [2,2,1] )

        for n in range(N):
            error = model.iterate(independentVariablesArray, dependantVariables, LR)
            errorArray.append(error)

        predictedArray = []
        for independentVariable in independentVariablesArray:
            predictedArray.append(model.feed_forward(independentVariable))
        rValue = calculate_r_squared(dependantVariables, predictedArray)

        plt.plot(errorArray)
        plt.show()
        print(rValue)


    #     def test_batch_learning(self):
    #         inputs = [
    #                 [0, 0],
    #                 [1, 0],
    #                 [0, 1],
    #                 [1, 1]
    #                 ]
    #         outputs = [ 0, 1, 1, 0 ]
    #         LR = 0.000000001
    #         N = 100000

    #         model = Model( [2,2,1] )

    #         for i in range(0, N):
    #             for idx in range(0, len(inputs)):
    #                 inputs = [0, 0]
    #                 outputs = [0]
    #                 model.iterate_training(inputs, outputs, LR)

    #                 inputs = [1, 0]
    #                 outputs = [1]
    #                 model.iterate_training(inputs, outputs, LR)

    #                 inputs = [0, 1]
    #                 outputs = [1]
    #                 model.iterate_training(inputs, outputs, LR)

    #                 inputs = [1, 1]
    #                 outputs = [0]
    #                 model.iterate_training(inputs, outputs, LR)

    #         self.assertEqual(model.feedforward([0,0]), 0)
    #         self.assertEqual(model.feedforward([1,0]), 1)
    #         self.assertEqual(model.feedforward([0,1]), 1)
    #         self.assertEqual(model.feedforward([1,1]), 0)

    #     def test_stochastic(self):
    #         inputs = [
    #                 [0, 0],
    #                 [1, 0],
    #                 [0, 1],
    #                 [1, 1]
    #                 ]
    #         outputs = [ 0, 1, 1, 0 ]
    #         LR = 0.000000001
    #         N = 100000

    #         model = Model( [2,2,1] )

    #         for i in range(0, N):
    #             for idx in range(0, len(inputs)):
    #                 inputs = [0, 0]
    #                 outputs = [0]
    #                 model.iterate_training(inputs, outputs, LR)


    #         for i in range(0, N):
    #             for idx in range(0, len(inputs)):
    #                 inputs = [1, 0]
    #                 outputs = [1]
    #                 model.iterate_training(inputs, outputs, LR)


    #         for i in range(0, N):
    #             for idx in range(0, len(inputs)):
    #                 inputs = [0, 1]
    #                 outputs = [1]
    #                 model.iterate_training(inputs, outputs, LR)


    #         for i in range(0, N):
    #             for idx in range(0, len(inputs)):
    #                 inputs = [1, 1]
    #                 outputs = [0]
    #                 model.iterate_training(inputs, outputs, LR)

    #         self.assertEqual(model.feedforward([0,0]), 0)
    #         self.assertEqual(model.feedforward([1,0]), 1)
    #         self.assertEqual(model.feedforward([0,1]), 1)
    #         self.assertEqual(model.feedforward([1,1]), 0)

    #     def test_video_lesson(self):
    #         model = Model( [2,2,1] )

    #         inputs = [5, 3]
    #         outputs = [12]

    #         input_neurons = []
    #         input_neurons.append(model.get_input_neuron(0))
    #         input_neurons.append(model.get_input_neuron(1))

    #         expected_output = outputs[0]
    #         input_neurons[0].bias = inputs[0]
    #         input_neurons[1].bias = inputs[1]

    #         # Starting from the hidden layer, each node input value is a weighted summation of all of its input.

    #         LR = 0.01
    #         predicted = expected_output

    #         ########################
    #         # feed the neural network with the input: the output of each node in the hidden
    #         # layers and the output layer is calculated
    #         ########################

    #         # feed forward
    #         for idx_layer in range(1, model.network.get_number_of_layers()):
    #             current_layer = model.network.get_layer(idx_layer)
    #             for idx_neuron in range(0, current_layer.get_number_of_neurons()):
    #                 current_layer.get_neuron(idx_neuron).calculate(idx_neuron)

    #         ########################
    #         # calculate the error, which is the estimated – the actual value
    #         ########################
    #         output_layer = model.network.get_output_layer()
    #         neuron = output_layer.get_neuron(0)
    #         actual    = neuron.bias

    #         error = (predicted - actual)

    #         ########################
    #         # backpropagate the error and calculate the derivative with the respect to each weight
    #         ########################

    #         output_o1 = output_layer.get_neuron(0).bias
    #         w5        = neuron.get_parent_neuron(0).weights[0]
    #         output_h1 = neuron.get_parent_neuron(0).bias
    #         output_h2 = neuron.get_parent_neuron(1).bias
    #         output_i1 = neuron.get_parent_neuron(0).get_parent_neuron(0).bias
    #         output_i2 = neuron.get_parent_neuron(0).get_parent_neuron(1).bias

    #         # w5
    #         # delta_w5 = error * [output_o1 * (1 - output_o1)] * output_h1
    #         delta_w5 = error * (output_o1 * (1 - output_o1)) * output_h1

    #         # w6
    #         # delta_w6 = error * [output_o1 * (1 - output_o1)] * output_h2
    #         delta_w6 = error * (output_o1 * (1 - output_o1)) * output_h2

    #         # w1
    #         # deltaE_w1 = error * [output_o1 * (1 - output_o1)] * w5
    #         deltaE_w1 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(0).weights[0]
    #         # delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1
    #         delta_w1 = deltaE_w1 * (output_h1 * (1 - output_h1)) * output_i1

    #         # w2
    #         # deltaE_w2 = error * [output_o1 * (1 - output_o1)] * w6
    #         deltaE_w2 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(1).weights[0]
    #         # delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1
    #         delta_w2 = deltaE_w2 * (output_h2 * (1 - output_h2)) * output_i1

    #         # w3
    #         # deltaE_w3 = error * [output_o1 * (1 - output_o1)] * w5
    #         deltaE_w3 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(0).weights[0]
    #         # delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2
    #         delta_w3 = deltaE_w3 * (output_h1 * (1 - output_h1)) * output_i2

    #         # w4
    #         # deltaE_w4 = error * [output_o1 * (1 - output_o1)] * w6
    #         deltaE_w4 = error * (output_o1 * (1 - output_o1)) * neuron.get_parent_neuron(1).weights[0]
    #         # delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2
    #         delta_w4 = deltaE_w4 * (output_h2 * (1 - output_h2)) * output_i2

    #         ###################
    #         # update each weight
    #         ###################

    #         # w5_new = w5_old - (LR * delta_w5)
    #         neuron.get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).weights[0] - (LR * delta_w5)
    #         # w6_new = w6_old - (LR * delta_w6)
    #         neuron.get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(1).weights[0] - (LR * delta_w6)

    #         # w1_new = w1_old - (LR * delta_w1)
    #         neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[0] - (LR * delta_w1)
    #         # w2_new = w2_old - (LR * delta_w2)
    #         neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(0).weights[1] - (LR * delta_w2)
    #         # w3_new = w3_old - (LR * delta_w3)
    #         neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[0] - (LR * delta_w3)
    #         # w4_new = w4_old - (LR * delta_w4)
    #         neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] = neuron.get_parent_neuron(0).get_parent_neuron(1).weights[1] - (LR * delta_w4)

    #         # model.draw()




    # def test_draw(self):
    #     model = self.setup_model()
    #     model.draw()

if __name__ == '__main__':
    unittest.main()

