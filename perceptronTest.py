import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib as mp
from perceptron import perceptron


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None) # extracting data from iris dataset
df = shuffle(df)

X = df.iloc[:, 0:4].values # X will be data used, in this case is length and width of sepal and petal for respective iris species
y = df.iloc[:,4].values # label will be type of iris species

train_data, test_data, train_labels, test_labels = train_test_split(X,y, test_size=0.25) # splitting dataset into test and train values, 25% for testing and 75% for training.

train_labels = np.where(train_labels == 'Iris-setosa', 1, -1) # Model will run to identify Iris-setosa species
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

perceptron = Perceptron(eta=0.1,n_iter=10) # training model
perceptron.fit(train_data, train_labels)

test_preds = perceptron.predict(test_data) # testing model
accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy: ', round(accuracy,2)*100, '%')

import matplotlib.pyplot as plt

plt.plot(range(1, len(errors) + 1), errors, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Training Errors per Epoch')

# Display the plot
plt.show()

errors = perceptron.errors
print(errors)
