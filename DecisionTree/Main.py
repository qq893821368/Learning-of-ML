from DecisionTree import tree
from kNN import kNN
from DecisionTree import treePlotter
"""


"""
data, labels = kNN.file2matrix('testSet.txt')
data = data.tolist()
for i in range(len(data)):
    data[i].append(labels[i])
    for j in range(2):
        if data[i][j] == 0:
            data[i][j] = 'y'
        else:
            data[i][j] = 'n'
labels = ['can fly', 'by oil', 'wheel number']
my_tree = tree.create_tree(data, labels)
treePlotter.create_plot(my_tree)