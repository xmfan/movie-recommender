from numpy import exp, array, random, dot
from neural_network_data import get_training_data_by_genre

# implementation details follow from my other repository: https://github.com/xmfan/neural-net-experiment/
class SingleLayerNeuralNetwork():
    def __init__(self, dimension):
        random.seed(1)

        # assign random weights to my inputs, a 3x1 matrix
        random_weights = random.random(dimension)

        # normalize weights over -1 to 1
        self.synaptic_weights = 2 * random_weights - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self,x):
        return x * (1 - x)

    def think(self, inputs):
        # activation function
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for _ in xrange(iterations):
            output = self.think(training_set_inputs)

            # loss function
            error = training_set_outputs - output

            # product of inputs with absolute error, gradiant descent
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # backpropagation
            self.synaptic_weights += adjustment

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        random_weights = random.random((number_of_inputs_per_neuron, number_of_neurons))
        self.synaptic_weights = 2 * random_weights - 1

class DeepLearningNeuralNetwork():

    # Use an invisible layer to learn combinations of A, B, C
    def __init__(self, layer1, layer2):
        self.layer1, self.layer2 = layer1, layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        layer1_output = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        layer2_output = self.__sigmoid(dot(layer1_output, self.layer2.synaptic_weights))
        return layer1_output, layer2_output

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for _ in xrange(iterations):
            layer1_output, layer2_output = self.think(training_set_inputs)

            layer2_error = training_set_outputs - layer2_output
            layer2_delta = layer2_error * self.__sigmoid_derivative(layer2_output)

            layer1_error = dot(layer2_delta, self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(layer1_output)

            layer1_adjustment = dot(training_set_inputs.T, layer1_delta)
            layer2_adjustment = dot(layer1_output.T, layer2_delta)

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment


if __name__ == "__main__":
    # single layer neural network
    print('Loading data...')
    action_training_data = get_training_data_by_genre('Action')
    training_input, training_output = action_training_data[0], action_training_data[1]
    formatted_training_output = array([training_output]).T
    print('Loading complete: %d rows' % len(action_training_data))

    single_layer_neural_network = SingleLayerNeuralNetwork((3,1))
    print "Random starting synaptic weights: "
    print single_layer_neural_network.synaptic_weights

    single_layer_neural_network.train(training_input, formatted_training_output, 60000)

    print "New synaptic weights after training: "
    print single_layer_neural_network.synaptic_weights

    # print "Considering output of new situation [1, 0, 0] -> ?: "
    # print neural_network.think(array([1, 0, 0]))

    # # deep learning
    # neural_network_data.get_movies_data_by_genre()
