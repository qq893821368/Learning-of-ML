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


if __name__ == '__main__':
    bayes.test_classify0()
