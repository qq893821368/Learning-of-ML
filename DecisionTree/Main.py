from DecisionTree import tree
from kNN import kNN
from DecisionTree import treePlotter

data, labels = kNN.file2matrix('testSets/testSet.txt', type='str')
data = data.tolist()
for i in range(len(data)):
    data[i].append(labels[i])
    for j in range(2):
        if data[i][j] == 1:
            data[i][j] = 'y'
        else:
            data[i][j] = 'n'
# labels = ['have house', 'have car', 'number of love']
labels = ['can fly', 'by oil', 'number of wheels']
my_tree = tree.create_tree(data, labels)
labels = ['can fly', 'by oil', 'number of wheels']
treePlotter.create_plot(my_tree)

print(tree.classify(my_tree, labels, ['y', 'y', 12]))


'''
myDat, labels = tree.create_data_set()
myTree = treePlotter.retrieve_tree(0)
print(treePlotter.classify(myTree, labels, [1, 0]))
print(treePlotter.classify(myTree, labels, [1, 1]))
'''