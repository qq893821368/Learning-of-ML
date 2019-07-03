from DecisionTree import tree
from kNN import kNN
from DecisionTree import treePlotter


# my test function
def my_test():
    data, labels = kNN.file2matrix('testSets/testSet.txt', type='str')
    data = data.tolist()
    for i in range(len(data)):
        data[i].append(labels[i])
        for j in range(2):
            if data[i][j] == 1:
                data[i][j] = 'y'
            else:
                data[i][j] = 'n'
    labels = ['can fly', 'by oil', 'number of wheels']
    my_tree = tree.create_tree(data, labels)
    labels = ['can fly', 'by oil', 'number of wheels']
    treePlotter.create_plot(my_tree)
    print(tree.classify(my_tree, labels, ['y', 'y', 12]))


# book's test function
def book_test():
    myDat, labels = tree.create_data_set()
    myTree = treePlotter.retrieve_tree(0)
    print(treePlotter.classify(myTree, labels, [1, 0]))
    print(treePlotter.classify(myTree, labels, [1, 1]))


def classify_glasses():
    fr = open('testSets/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    lenses_tree = tree.create_tree(lenses, lenses_labels)
    print('lenses_tree:', lenses_tree)
    treePlotter.create_plot(lenses_tree)
    tree.store_tree(lenses_tree, 'trees_repository/tree_for_lenses.txt')


def classify_dating():
    fr = open('testSets/dating.txt')
    dating = [inst.strip().split('\t') for inst in fr.readlines()]
    dating_labels = ['age', 'income', 'height']
    dating_tree = tree.create_tree(dating, dating_labels)
    print(dating_tree)
    treePlotter.create_plot(dating_tree)


def test_dating():
    fr = open('testSets/dating_test.txt')
    dating = [inst.strip().split('\t') for inst in fr.readlines()]
    dating_labels = ['height', 'income', 'age']
    for line in dating:
        del(line[-1])
    print('dating:')
    for line in dating:
        print(line)
    print('entropy:', tree.calculate_shannon_entropy(dating))

    fr = open('testSets/dating_test2.txt')
    dating = [inst.strip().split('\t') for inst in fr.readlines()]
    dating_labels = ['height', 'income', 'age']
    for line in dating:
        del (line[-1])
    print('dating:')
    for line in dating:
        print(line)
    print('entropy:', tree.calculate_shannon_entropy(dating))

    fr = open('testSets/dating.txt')
    dating = [inst.strip().split('\t') for inst in fr.readlines()]
    dating_labels = ['height', 'income', 'age']
    for line in dating:
        del (line[-1])
    print('dating:')
    for line in dating:
        print(line)
    print('entropy:', tree.calculate_shannon_entropy(dating))


def test_fake():
    fr = open('testSets/fake_set.txt')
    fake = [inst.strip().split('\t') for inst in fr.readlines()]
    fake_labels = ['number']
    fake_tree = tree.create_tree(fake, fake_labels)
    print(fake_tree)


if __name__ == '__main__':
    # test_fake()
    # classify_glasses()
    # my_test()
    classify_dating()
    # test_dating()


