# Solving IRIS flower challenge using a Neural Network.

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. 

### The challenge.
Deploy a Neural Network which will analyze the four features of an IRIS flower and predict it's specie. We will use tensorflow to deploy the neural network.

### The dataset.

The IRIS dataset has five coloumns: Sepal-Length, Sepal-Width, Petal-Length, Petal-Width, and specie.
It has 150 entries (rows).

Sepal-Length, Sepal-Width, Petal-Length, and Petal-Width are numerical in nature.
while, Specie is categoraical. A particular flower can be any one of these three species: Iris-setosa, Iris-versicolor, Iris-virginica.

The dataset is divided into Training and testing sets and they contain 105 and 45 entries respectively.

```
test_size = 0.30
seed = 7
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=test_size,random_state=seed)
```

### Prerequisites

* Python 3.5.x
* Tensorflow
* Numpy
* Pandas
* SKlearn
* Jupyter Notebook (optional)


### Structure of the Neural Network.

It is a three layered neural network with 4 neurons in the input layer, 8 neurons in the hidden layer, and 3 neurons in the output layer.

![image](https://user-images.githubusercontent.com/16969678/36962050-f0503a2a-2073-11e8-8358-a749a84f3c5b.png)
)

This arrangement predicts the species with 96% accuracy.

