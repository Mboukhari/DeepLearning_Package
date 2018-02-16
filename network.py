# -*- coding: utf-8 -*-
"""
@date: 2017/09/10
@version: 0.0.1
@author: AGAMBO & CFANG & MBOUKHARI
"""

import numpy as np
import math

       
class Sequential(object):
    
    def __init__(self):
        self.layer = []
        self.layer_id = 0
        
    def add_layer(self, new_layer): 
        """
        This function take neural network layer as input and return neural network with sequentials layers.
        # Example:
        ```python
        # initialise neural network
        from layers import Layers
        
        model = Sequential()
        l = Layers()
        
        # create network layer
        layer_1 = l.dense_layer(output_dim=2, input_dim=3, init='uniform', activation='sigmoid')
        layer_2 = l.dense_layer(output_dim=1, input_dim=2, init='uniform', activation='sigmoid')
        
        # add layer into neural network
        model.add_layer(layer_1)
        model.add_layer(layer_2)
        
        # show neural network
        network = model.get_network()
        print network        
        ```       
        """
        
        self.layer.append({'layer_id':self.layer_id, 'layer':new_layer}) 
        self.layer_id += 1
        
    def get_network(self):
        return list(self.layer)


class ConvNet(object):
    pass


class LSTM(object):
    pass



