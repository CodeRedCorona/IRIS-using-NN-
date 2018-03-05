#DEPENDENCIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import model_selection

#Handles.
hidden_layer_nodes = 8

#LOADING DATA.
filename = "Irisdata.csv"
column_names = ['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width','Class']
dataframe = pd.read_csv(filename, names=column_names)

#One Hot Encoding.
Class_dummies = pd.get_dummies(dataframe['Class'], prefix = 'Class')
dataframe = pd.concat([dataframe, Class_dummies], axis = 1)
dataframe.drop('Class', axis = 1, inplace = True)

#Creating tensors.
features = dataframe.loc[:, ['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width']].as_matrix()
labels = dataframe.loc[:, ['Class_Iris-setosa','Class_Iris-versicolor', 'Class_Iris-virginica']].as_matrix()

#Shuffling data .
#Creating training and test data set. 
test_size = 0.30
seed = 7
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=test_size,random_state=seed)


#Defining Hyper-parameters. 
#Hyper-parameters will control how fast out NN will converge.
learning_rate =  0.0001
training_epochs = 500
display_step = 50
n_samples = labels_train.size

#DEFINING NN 

#Layer 1 - Input Layer.
#None means any number of examples. 
input = tf.placeholder(tf.float32, [None, 4])
weights1 = tf.Variable(tf.random_normal([4,hidden_layer_nodes]))
biases1 = tf.Variable(tf.random_normal([hidden_layer_nodes]))
output1 = tf.nn.relu(tf.add(tf.matmul(input, weights1), biases1))

#Layer 2 - Hidden Layer.
weights2 = tf.Variable(tf.random_normal([hidden_layer_nodes,3]))
biases2 = tf.Variable(tf.random_normal([3]))
output2 = tf.add(tf.matmul(output1, weights2), biases2)

#Layer 3 - Output Layer.
#softmax() aka Sigmoid function is the activation function.
#It is applied to the values we just calculated.
output = tf.nn.softmax(output2)
#Matrix of labels or output matrix.
output_ = tf.placeholder(tf.float32, [None, 3])

#TRAINING NN
cost = tf.reduce_mean(-tf.reduce_sum(output_ * tf.log(output), axis = 0))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init =  tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range (training_epochs):
	sess.run(optimizer, feed_dict = {input:features_train, output_: labels_train})
	if(i)%display_step == 0:
		cc = sess.run(cost, feed_dict = {input:features_train, output_ : labels_train})
		print ("training_epoch: ", '%04d' % (i), "cost=", "{:.9f}".format(cc))
#Making predictions.		
pred = sess.run(output, feed_dict = {input:features_test})


#The specieIdentifier function converts the one hot encoded data back into the specie name.
def specieIdentifier(dummy):
	if dummy[0] == 1: return "Iris-setosa"
	elif dummy[1] == 1: return "Iris-versicolor"
	elif dummy[2] == 1: return "Iris-virginica"
	else: return "error"

#Printing serial no., Actual species, and the prediction. 
for i in range(len(features_test)):
	print(i+ 1,"\nActual   : ", specieIdentifier(labels_test[i]), "\npredicted: ", specieIdentifier(np.rint(pred[i])), "\n")

