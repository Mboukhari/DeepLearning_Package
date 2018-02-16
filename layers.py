# -*- coding: utf-8 -*-
"""
@date: 2017/09/10
@version: 0.0.1
@author: AGAMBO & CFANG & MBOUKHARI
"""

import numpy as np
import random

random.seed(1)

class Layers(object):
    
    def normal(self, n):
        """
        This function return normal distribution vector which contains n float values between 0 and 1.
        # Exemple:
        ```python
        l = Layers()
        output = l.normal(5)
        print output
        ```      
        """ 
        return np.random.normal(0, 0.5, n)
    
    def random(self, n):
        """
        This function return random vector which contains n float values between 0 and 1.
        # Exemple:
        ```python
        l = Layers()
        output = l.random(5)
        print output
        ```      
        """
        return [random.random() for i in range(n)]        
    
    def uniform(self, n):
        """
        This function return uniform distribution vector which contains n float values between 0 and 1.
        # Exemple:
        ```python
        l = Layers()
        output = l.uniform(5)
        print output
        ```      
        """
        return np.random.uniform(0, 1, n)
    
    def dense_layer(self, output_dim, input_dim, init, activation):
        """
        This function return a dictionnary of input arguments and associated value.
        It also add "weights" item and a list of value associated to weight. 
        The list values come from init parameters.  
        # Exemple:
        ```python
        l = Layers()
        output = l.dense_layer(output_dim=2, input_dim=3, init='random', activation='sigmoid')
        print output
        ```       
        """        
        output = {}
        for i in range(output_dim):
            if init == 'uniform':
                weights = list(self.uniform(input_dim + 1))
            elif init == 'normal':
                weights = list(self.normal(input_dim + 1))
            elif init == 'random':
                weights = list(self.random(input_dim + 1))
            output[i] = {'weights':weights,
                  'delta_weights':[k * 0 for k in range(input_dim + 1)],
                  'delta_neuron':0,
                  'output_dim':output_dim,
                  'input_dim':input_dim,
                  'init':init,
                  'activation':activation}       
        return output
        
    def conv2D_layer(self, output_dim, input_dim, init, activation):
        pass
    
    def conv3D_layer(self, output_dim, input_dim, init, activation):
        pass


