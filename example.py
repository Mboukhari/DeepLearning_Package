# -*- coding: utf-8 -*-
"""
@date: 2017/09/10
@version: 0.0.1
@author: AGAMBO & CFANG
"""



import sys
sys.path.append('/home/agambo/analytics/smart/analyse_campaign/VNNet')

from layers import Layers
from network import Sequential
from models import VNNetMulticlassClassifier


# 0- Load dataset
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

X = [x[:-1] for x in dataset]
Y = [x[-1] for x in dataset]


# 1- initialise neural network
neural_network = Sequential()

# 2- create network layer
l = Layers()
layer_1 = l.dense_layer(output_dim=2, input_dim=2, init='random', activation='sigmoid')
layer_2 = l.dense_layer(output_dim=2, input_dim=2, init='random', activation='sigmoid')
print layer_1 

# 3- add layer into neural network
neural_network.add_layer(layer_1)
neural_network.add_layer(layer_2)
network = neural_network.get_network()
print network
   
# 4- compile model
model = VNNetMulticlassClassifier(network=network, loss='categorical_crossentropy', optimizer='sgd', l_rate=0.5, metrics=['auc'])

# 5- Train model
model.fit(X, Y, nb_epoch=200, batch_size=1, nb_class=2)

# 6- Evaluate model
model.evaluate(X, Y)
print("AUC: %.1f%%" % (model.score * 100))

# 7- Predict new value
prediction = model.predict([[2.7810836,2.550537003]])
print prediction

# 8- save model and weignt
model.save_weights('/home/agambo/analytics/smart/analyse_campaign/data/wieghts_saved.json')

# 9- load saved model
loaded_network = model.load_weights('/home/agambo/analytics/smart/analyse_campaign/data/wieghts_saved.json')


# 10- compile saved model
loaded_model = VNNetMulticlassClassifier(network=loaded_network, loss='categorical_crossentropy', optimizer='sgd', l_rate=0.5, metrics=['auc'])

# 11- make prediction with loaded model
prediction = loaded_model.predict([[1.38807019,1.850220317]])
print prediction

# 12- get proba 
proba = loaded_model.predict([[1.38807019,1.850220317]], func='proba')
print proba