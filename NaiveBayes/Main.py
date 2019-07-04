import NaiveBayes.bayes as bayes

data, labels = bayes.load_data_set()
my_vocab_list = bayes.create_vocab_list(data)
train_matrix = []
'''
print('vocab_list:')
i = 1
for word in my_vocab_list:
    print(word, end=", ")
    if i % 8 == 0:
        print()
    i += 1
print()
'''
for post in data:
    train_matrix.append(bayes.set_of_words2vec(my_vocab_list, post))

# dirty = ['stupid', 'fuck', 'fuck', 'fuck you']
# print(bayes.create_vocab_list([dirty]))
# print(len(train_matrix) == len(labels))
print('\nvocab_list:', my_vocab_list)
p0, p1, pAb = bayes.train_naive_bayes0(train_matrix, labels)




