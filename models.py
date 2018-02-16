# -*- coding: utf-8 -*-
"""
@date: 2017/09/10
@version: 0.0.1
@author: AGAMBO & CFANG
"""


import numpy as np
import json
import math


class Base(object):

    def sigmoid(self, activation):
        """
        This function take a scalar value to return non-negative value between 0 and 1
        ...
        # Exemple:
        ```python
        base = Base()
        output = base.sigmoid(12)
        print output
        ```
        """
        return 1.0 / (1.0 + math.exp(-activation))
        
    def relu(self, activation):
        """
        This function take a scalar value to return 0 if input value is less than 0. It returns input value if not. 
        ...
        # Exemple:
        ```python
        base = Base()
        output = base.relu(-0.12)
        print output
        ```
        """
        return activation if activation > 0 else 0
        
    def sigmoid_derivative(self, output):
        """
        This function take a scalar value to return a sigmoid derivative value.
        ...
        # Exemple:
        ```python
        base = Base()
        output = base.sigmoid_derivative(0.75)
        print output
        ```
        """
        return output * (1.0 - output)
        
    def activate(self, weights, inputs):
        """
        This function return sum of n-wise element product given two input vectors.
        The second vector have (first vector - 1) element. So the last element of first vector will be multiply by 1. 
        ...
        # Exemple:
        ```python
        weights = [0.13436424411240122, 0.8474337369372327, 0.763774618976614]
        inputs = [1, 0]
        output = activate(weights, row)
        print output
        ```
        """
        return sum([w * i for w, i in zip(weights[:-1], inputs)]) + weights[-1]
        
    def forward_propagate(self, network, row):
        """
        This function apply network weight on input value to provide neurone output value.
        # Example:
        ```python
        network = [
            {'layer': {
                0: {'activation': 'sigmoid',
                'delta_weights': [0, 0, 0, 0],
                'init': 'uniform',
                'input_dim': 3,
                'output_dim': 2,
                'weights': [0.93120601968902172,  0.024899227550348013, 0.60054891746412253, 0.95012950041364563]},
               1: {'activation': 'sigmoid',
                'delta_weights': [0, 0, 0, 0],
                'init': 'uniform',
                'input_dim': 3,
                'output_dim': 2,
                'weights': [0.2303028790209648, 0.54848991923603041, 0.90912837488673126, 0.13316944575925016]}},
            'layer_id': 0},
            {'layer': {
                0: {'activation': 'sigmoid',
                'delta_weights': [0, 0, 0],
                'init': 'uniform',
                'input_dim': 2,
                'output_dim': 1,
                'weights': [0.52341258067376584, 0.75040985910203484, 0.66901324088391378]}},
            'layer_id': 1}]
        row = [0, 1, 0]        
        b = Base()
        output = b.forward_propagate(network, row)
        print output
        ```
        """
        network = network
        inputs = row
        input_layer = {}
        i = 0
        input_layer[i] = inputs
        for layer in range(len(network)):
            layer_data = network[layer]
            current_inputs = input_layer[i]
            new_input = []
            i+=1
            for neuron_id in layer_data['layer']:
                neuron_weights = layer_data['layer'][neuron_id]['weights']
                activation = self.activate(neuron_weights, current_inputs)
                activation_f = layer_data['layer'][neuron_id]['activation']
                if activation_f == 'sigmoid':
                    activation = self.sigmoid(activation)
                elif activation_f == 'relu':
                    activation = self.relu(activation)
                layer_data['layer'][neuron_id]['neuron_output'] = activation
                new_input.append(activation)
                input_layer[i] = new_input
        output = input_layer[i]

        return output
    
    def backward_propagate_error(self, network, expected, row):
        """
        This function compute retro-propagation to compute neuron and weights delta to provide values for weights update. 
        ```python
        network = [
        {'layer': {
            0: {'activation': 'sigmoid',
            'delta_weights': [0, 0, 0, 0],
            'init': 'uniform',
            'input_dim': 3,
            'neuron_output': 0.7261206944117606,
            'output_dim': 2,
            'weights': [0.9312060196890217, 0.024899227550348013, 0.6005489174641225, 0.9501295004136456]},
           1: {'activation': 'sigmoid',
            'delta_weights': [0, 0, 0, 0],
            'init': 'uniform',
            'input_dim': 3,
            'neuron_output': 0.664108949795203,
            'output_dim': 2,
            'weights': [0.2303028790209648, 0.5484899192360304, 0.9091283748867313, 0.13316944575925016]}},
          'layer_id': 0},
         {'layer': {
             0: {'activation': 'sigmoid',
            'delta_weights': [0, 0, 0],
            'init': 'uniform',
            'input_dim': 2,
            'neuron_output': 0.8245419230676196,
            'output_dim': 1,
            'weights': [0.5234125806737658, 0.7504098591020348, 0.6690132408839138]}},
          'layer_id': 1}]
        expected = [0]
        row = [0, 1, 0]
        b = Base()
        b.backward_propagate_error(network, expected, row)
        print network
        ```
        """
        for layer in reversed(range(len(network))):
            layer_data = network[layer]
            errors = []
            if layer != len(network)-1: # hidden or input layer
                for neuron_id in range(len(layer_data['layer'])):
                    error = 0.0
                    for next_layer_neuron_id in range(len(network[layer + 1]['layer'])):
                        neuron = network[layer + 1]['layer'][next_layer_neuron_id]
                        error += (neuron['weights'][neuron_id] * neuron['delta_neuron'])
                    errors.append(error)
            else: # output layer
                for neuron_id in range(len(layer_data['layer'])):
                    neuron_output = layer_data['layer'][neuron_id]['neuron_output']
                    errors.append(expected[neuron_id] - neuron_output)

            for neuron_id in range(len(layer_data['layer'])):
                neuron = layer_data['layer'][neuron_id]
                neuron['delta_neuron'] = errors[neuron_id] * self.sigmoid_derivative(neuron['neuron_output'])

            for neuron_id in range(len(layer_data['layer'])):
                if layer != 0: # hidden or output layer
                    inputs = [network[layer - 1]['layer'][preview_neuron_id]['neuron_output'] for preview_neuron_id in network[layer - 1]['layer']]
                else: # input layer
                    inputs = row
                for neuron_input_id in range(len(inputs)):
                    layer_data['layer'][neuron_id]['delta_weights'][neuron_input_id] += layer_data['layer'][neuron_id]['delta_neuron'] * inputs[neuron_input_id]
                layer_data['layer'][neuron_id]['delta_weights'][-1] += (layer_data['layer'][neuron_id]['delta_neuron'] * 1.0)
    
    def update_weights(self, network, l_rate, batch_size):
        """
        This function take neuron weight delta to update weights
        ```python
        network = [
            {'layer': {
                0: {'activation': 'sigmoid',
                'delta_neuron': -0.012416838662802198,
                'delta_weights': [0.0, -0.012416838662802198, 0.0, -0.012416838662802198],
                'init': 'uniform',
                'input_dim': 3,
                'neuron_output': 0.7261206944117606,
                'output_dim': 2,
                'weights': [0.9312060196890217, 0.024899227550348013, 0.6005489174641225, 0.9501295004136456]},
               1: {'activation': 'sigmoid',
                'delta_neuron': -0.019968026542664958,
                'delta_weights': [0.0, -0.019968026542664958, 0.0, -0.019968026542664958],
                'init': 'uniform',
                'input_dim': 3,
                'neuron_output': 0.664108949795203,
                'output_dim': 2,
                'weights': [0.2303028790209648, 0.5484899192360304, 0.9091283748867313, 0.13316944575925016]}},
              'layer_id': 0},
             {'layer': {
                 0: {'activation': 'sigmoid',
                'delta_neuron': -0.11928857448814481,
                'delta_weights': [-0.08661790254272074, -0.0792206099258887, -0.11928857448814481],
                'init': 'uniform',
                'input_dim': 2,
                'neuron_output': 0.8245419230676196,
                'output_dim': 1,
                'weights': [0.5234125806737658, 0.7504098591020348, 0.6690132408839138]}},
              'layer_id': 1}]
        l_rate = 0.5
        batch_size = 1
        b = Base()
        b.update_weights(network, l_rate, batch_size)
        print network
        ```
        """
        for layer in range(len(network)):
            layer_data = network[layer]
            for neuron_id in range(len(layer_data['layer'])):
                for weight_id in range(len(layer_data['layer'][neuron_id]['weights'])):
                    # update weight
                    layer_data['layer'][neuron_id]['weights'][weight_id] += l_rate * (layer_data['layer'][neuron_id]['delta_weights'][weight_id])/batch_size
                    # initialise delta_weight
                    layer_data['layer'][neuron_id]['delta_weights'][weight_id] = 0


class VNNetRegressor(Base):
    
    pass
        

class VNNetBinnaryClassifier(Base):
    
    def __init__(self, network, loss, optimizer, l_rate, metrics, verbose=True):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.l_rate = l_rate
        self.metrics = metrics
        self.score = None
        self.error_diagnostic = dict()
        self.error = None
        self.verbose = verbose
        
    def fit(self, X, Y, nb_epoch, batch_size):
        """
        This function train model to get parameters with minimize prediction error.
        """
        for epoch in range(nb_epoch):
            sum_error = 0
            batch_counter = 0
            index = 0
            for row in X:
                outputs = self.forward_propagate(self.network, row)
                expected = [Y[index]]
                sum_error += (expected[0] - outputs[0])**2
                self.backward_propagate_error(self.network, expected, row)
                batch_counter+=1
                index+=1
                if batch_counter == batch_size:
                    self.update_weights(self.network, self.l_rate, batch_size)
                    batch_counter = 0
            self.error_diagnostic[epoch] = sum_error
            if self.verbose == True:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))

        self.error = sum_error
        
    def predict(self, X, alpha=0.5):
        """
        This function take input parameters to make prediction.
        """
        outputs = []
        for row in X:
            output = self.forward_propagate(self.network, row)
            output = 0 if output[0] < alpha else 1
            outputs.append(output)
        return outputs
        
    def evaluate(self, X, Y):
        """
        This function evaluate model error.
        """
        preds = self.predict(X)
        score = 0.0
        for i in range(len(Y)):
            if preds[i] == Y[i]:
                score += 1.0
        score = score / float(len(Y))
        self.score = score
        return score
        
    def save_weights(self, file_path):
        """
        This function save model weight into JSON file.
        """
        f = open(file_path, 'wb')
        f.write(json.dumps(self.network))
        f.close()

    def load_weights(self, file_path):
        """
        This function load model weight.
        """
        f = open(file_path, 'rb')
        return json.loads(f.readline())


class VNNetMulticlassClassifier(Base):
    
    def __init__(self, network, loss, optimizer, l_rate, metrics, verbose=True):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.l_rate = l_rate
        self.metrics = metrics
        self.score = None
        self.error_diagnostic = dict()
        self.error = None
        self.verbose = verbose
        
    def fit(self, X, Y, nb_epoch, batch_size, nb_class):
        """
        This function train model to get parameters with minimize prediction error.
        """
        for epoch in range(nb_epoch):
            sum_error = 0
            batch_counter = 0
            index = 0
            for row in X:
                outputs = self.forward_propagate(self.network, row)
                expected = [0 for i in range(nb_class)]
                expected[Y[index]] = 1
                sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(self.network, expected, row)
                batch_counter+=1
                index+=1
                if batch_counter == batch_size:
                    self.update_weights(self.network, self.l_rate, batch_size)
                    batch_counter = 0
            self.error_diagnostic[epoch] = sum_error
            if self.verbose == True:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))

        self.error = sum_error
        
    def predict(self, X, func='softmax'):
        """
        This function take input parameters to make prediction.
        """
        outputs = []
        for row in X:
            output = self.forward_propagate(self.network, row)
            if func == 'softmax':                
                outputs.append(output.index(max(output)))
            elif func == 'proba':
                outputs = output
        return outputs
        
    def evaluate(self, X, Y):
        """
        This function evaluate model error.
        """
        preds = self.predict(X)
        score = 0.0
        for i in range(len(Y)):
            if preds[i] == Y[i]:
                score += 1.0
        score = score / float(len(Y))
        self.score = score
        return score
        
    def save_weights(self, file_path):
        """
        This function save model weight into JSON file.
        """
        f = open(file_path, 'wb')
        f.write(json.dumps(self.network))
        f.close()
        
    def load_weights(self, file_path):
        """
        This function load model weight.
        """
        f = open(file_path, 'rb')
        return json.loads(f.readline())
