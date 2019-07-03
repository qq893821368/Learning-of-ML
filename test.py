import DecisionTree.tree as tree
import DecisionTree.treePlotter as tplt

fr = open('DecisionTree/testSets/fake_set.txt')
fake = [inst.strip().split('\t') for inst in fr.readlines()]
print('fake:')
fake_labels = ['number']
for line in fake:
    print(line)
fake_tree = tree.create_tree(fake, fake_labels)
print(fake_tree)
tplt.create_plot(fake_tree)