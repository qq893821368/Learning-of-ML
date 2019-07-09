import NaiveBayes.bayes as bayes
from math import e


def my_test():
    my_vocab_list = ['fuck', 'shit', 'silly', 'bullshit', 'slut', 'bitch']
    my_doc = [['fuck'], ['a'], ['c'], ['fuck']]
    my_labels = [1, 0, 0, 1]
    my_mat = []
    for d in my_doc:
        my_mat.append(bayes.set_of_words2vec(my_vocab_list, d))

    p0, p1, pAb = bayes.train_naive_bayes0(my_mat, my_labels)
    vec = bayes.set_of_words2vec(my_vocab_list, ['fuck'])
    print('classify1: ', bayes.classify_naive_bayes1(vec, p0, p1, pAb))
    print('classify0: ', bayes.classify_naive_bayes0(vec, p0, p1, pAb))


def book_test():
    data, labels = bayes.load_data_set()
    my_vocab_list = bayes.create_vocab_list(data)
    train_matrix = []
    for document in data:
        train_matrix.append(bayes.set_of_words2vec(my_vocab_list, document))
    p0, p1, pAb = bayes.train_naive_bayes0(train_matrix, labels)
    vec = bayes.set_of_words2vec(my_vocab_list, ['I', 'am', 'a', 'handsome', 'boy'])
    print('classify1: ', bayes.classify_naive_bayes1(vec, p0, p1, pAb))
    print('classify0: ', bayes.classify_naive_bayes0(vec, p0, p1, pAb))


def test_bag_func():
    vocab_list = ['I', 'You', 'you', 'my', 'your']
    input_set = [['I', 'am', 'your', 'father'],
                 ['Oh', 'no', 'It', 'is', 'imposable'],
                 ['You', 'do', 'not', 'need', 'to', 'be', 'surprised']]
    print('vocab list is : ', vocab_list)
    for vec in input_set:
        print(bayes.bag_of_words2vector_model(vocab_list, vec))


def new_book_test():
    data, labels = bayes.load_data_set()
    my_vocab_list = bayes.create_vocab_list(data)
    train_matrix = []
    for document in data:
        train_matrix.append(bayes.bag_of_words2vec(my_vocab_list, document))
    p0, p1, pAb = bayes.train_naive_bayes0(train_matrix, labels)
    vec = bayes.set_of_words2vec(my_vocab_list, ['I', 'am', 'a', 'handsome', 'boy'])
    t = 1
    print('vocab_list: ')
    for i in range(len(my_vocab_list)):
        print(my_vocab_list[i], end=", ")
        if t % 8 == 0:
            print()
        t += 1
    print()
    print('train_matrix:')
    for line in train_matrix:
        print(line)
    print('classify0: ', bayes.classify_naive_bayes0(vec, p0, p1, pAb))


if __name__ == '__main__':
    bayes.spam_test_by_bag()
