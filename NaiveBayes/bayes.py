from numpy import *


def load_data_set():
    """
    创建一个默认的数据集
    :return: 文本列表, list
    :return: 词条向量, list, 表示所给的文本列表的对应项是否出现侮辱性词语
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage', ],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


def create_vocab_list(data_set):
    """
    给定一个数据集, 返回其中的词汇集合列表
    数据集为矩阵形式, 至少为[['word 1', 'word 2', ... , 'word n']]
    因为默认数据集是一个文档的集合, 即下图形式:
        ┌                       ┐
                [document 1]
                [document 2]           [document n] => ['word 1', 'word 2', ... , 'word n']
                [..........]
                [document n]
        └                       ┘,
    :param data_set: 数据集, list形式, 文档列表
    :return: 词汇集合列表, list形式
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 用或符号对两个集合进行并集运算
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    给定词汇集合和文本, 计算词条向量, 即是否出现某侮辱性词语
    简而言之是检查input_set在vocab_list中的出现情况
    :param vocab_list: 词汇集合, list形式, 侮辱性词汇集合 or 文档词汇集合
    :param input_set: 输入文档, set形式或list形式, 待分类文本
    :return: 词条向量, list, 表示所给的vocab_list的对应词汇是否出现
    """
    return_vector = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] = 1
        else:
            pass  # print语句太冗余, 暂时删除
            # print('the word: %s is not in my vocabulary!' % word)
    return return_vector


def train_naive_bayes0(train_matrix, train_category):
    """
    给定一个描述文档列表中各文档的词汇出现情况的矩阵, 和一个文档类别列表,
    返回是否侮辱性文档的条件下, 每个词汇出现的概率以及侮辱性文档概率,
    即求的bayes formula中的p(A)、p(B|A), 这里指p(ci)以及p(ω|ci)
    bayes formula : p(A|B) = (p(B|A)·p(A))/p(B)
    可用于计算出现每个词汇的条件下, 是否侮辱性文档的概率
    train_matrix如下图所示:
        ┌                       ┐
                [doc_voc 1]
                [doc_voc 2]             [doc_voc n] => [0, 0, 1, ..., 1] (0, 1根据词汇出现情况变化而变化)
                [..........]            [doc_voc n].size和词汇集合的大小相同
                [doc_voc n]
        └                       ┘
    train_category如下图所示:
        [0, 1, ..., 0] (0, 1根据文档类型变化而变化)
    train_category.size和train_matrix.size相同
    :param train_matrix: 文档的词汇出现情况矩阵, list形式, 描述每个文档中出现词汇集合中的词汇的情况
    :param train_category: 文档类别列表, list形式, 描述每个文档是否侮辱性文档
    :return:非侮辱性文档条件下各词汇出现概率, numpy.array
    :return:侮辱性文档条件下各词汇出现概率, numpy.array
    :return:侮辱性文档概率, float
    """
    num_train_docs = len(train_matrix)  # 文档个数
    num_words = len(train_matrix[0])  # 词汇个数
    p_abusive = sum(train_category) / float(num_train_docs)  # 侮辱性文档概率
    p0_denom = 2.0  # p0分母
    p1_denom = 2.0  # p1分母
    p0_num = ones(num_words)  # p(词汇i出现|非侮辱性文档)
    p1_num = ones(num_words)  # p(词汇i出现|侮辱性文档)
    for i in range(num_train_docs):  # 每个文档
        if train_category[i] == 1:  # 如果文档是侮辱性文档
            p1_num += train_matrix[i]  # 在侮辱性文档的词汇概率矩阵中记录
            p1_denom += sum(train_matrix[i])  # 记录词汇集合中的所有词汇的出现总次数, 上句为分子, 本句为分母, 结尾相除可的概率
        else:  # 如果不是侮辱性文档
            p0_num += train_matrix[i]  # 在非侮辱性文档的词汇概率矩阵中记录
            p0_denom += sum(train_matrix[i])  # 同上, 记录
    p1_vector = log(p1_num / p1_denom)  # 相除求得侮辱性文档的词汇出现次数/词汇集合中的词汇出现的总次数, 即在侮辱性文档的词汇概率矩阵中记录
    p0_vector = log(p0_num / p0_denom)  # 同上, 求得在非侮辱性文档的词汇概率矩阵中记录

    # # 这里注意, 无论什么文档下, 各个词汇出现的概率之和并不为1, 只是因为load_data_set中的假数据
    # # 在经过了书中的操作后, 分母为所有词汇出现次数总和, 所以最终概率和为1
    return p0_vector, p1_vector, p_abusive  # 返回(非)侮辱性文档条件下各词汇出现概率, (非)侮辱性文档的概率


def classify_naive_bayes0(vector2classify, p0_vector, p1_vector, p_class1):
    """
    给定一条由文档经过set_of_words2vec生成的表示待分类文档的词汇集合中词汇出现情况的待分类向量,
    以及是否侮辱性文档的条件下, 词汇集合中各词汇出现的概率的向量, 和是侮辱性文档的概率,
    返回该向量代表的文档属于的类型
    :param vector2classify: 待分类向量, list形式或numpy.array形式, set_of_words2vec可生成, 文档中各词汇出现情况向量
    :param p0_vector: 非侮辱性文档条件下各词汇出现概率
    :param p1_vector: 侮辱性文档条件下各词汇出现概率
    :param p_class1: 是侮辱性文档的概率
    :return: 是否侮辱性文档, bool
    """
    # # naive bayes只要求比较各个概率的大小, 而公式中的分母在各个概率计算中相同,
    # # 只起到求得具体数值的作用, 故忽略分母p(ω)
    p1 = sum(vector2classify * p1_vector) + log(p_class1)
    # 这里原本应该求∏(vector2classify * pi_vector) * p(ci) [vector2classify中为0的相乘项按1处理, 下同]
    # 由于概率均小于1, 为了防止下溢出, 求log(∏(vector2classify * pi_vector) * p(ci)),
    # 等价于log(∏(vector2classify * pi_vector)) + log (p(ci)), log(ab) = log a + log b
    # 又因为pi_vector经过了log处理, 故又等价于sum(vector2classify * p1_vector) + log (p(ci))
    p0 = sum(vector2classify * p0_vector) + log((1.0 - p_class1))
    # 由于函数的单调性不受log符号影响, 所以依然可以采用p1, p0比较大小的方式来判断类别
    if p1 > p0:
        return True
    else:
        return False


def test_classify0():
    """
    测试函数, 测试classify_naive_bayes0方法和自己写的classify_naive_bayes1方法,
    除了预设数据还提供一次用户自定数据的机会, 来对两个方法进行测试
    """
    data, labels = load_data_set()
    my_vocab_list = create_vocab_list(data)
    train_mat = []
    for document in data:
        train_mat.append(set_of_words2vec(my_vocab_list, document))
    p0_vector, p1_vector, p_Ab = train_naive_bayes0(train_mat, labels)
    test_entry = ['love', 'my', 'dalmatian']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))  # 可以不用array()
    print(test_entry, ' classified as: ', classify_naive_bayes0(this_doc, p0_vector, p1_vector, p_Ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, ' classified as: ', classify_naive_bayes0(this_doc, p0_vector, p1_vector, p_Ab))
    input_str = input('input a sentence to classify:')
    test_entry = input_str.split()
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, ' classified as: ', classify_naive_bayes1(this_doc, p0_vector, p1_vector, p_Ab))


def classify_naive_bayes1(vector2classify, p0_vector, p1_vector, p_class1):
    """
    完整bayes formula下的分类方法
    :param vector2classify: 同classify_naive_bayes0
    :param p0_vector: 同classify_naive_bayes0
    :param p1_vector: 同classify_naive_bayes0
    :param p_class1: 同classify_naive_bayes0
    :return: 同classify_naive_bayes0
    """
    p0_vector = e**p0_vector  # 还原log
    p1_vector = e**p1_vector
    p_w_c0 = sum(p0_vector * vector2classify)  # 求p(ωi|ci)之和
    p_w_c1 = sum(p1_vector * vector2classify)
    p_w = ones(len(vector2classify)) / len(vector2classify)  # 求p(ωi)
    p0 = p_w_c0 * (1.0 - p_class1) / sum(p_w * vector2classify)  # 求p(ωi|ci)*p(ci)/p(ωi)之和
    p1 = p_w_c1 * p_class1 / sum(p_w * vector2classify)
    if p1 > p0:
        return True
    else:
        return False


def bag_of_words2vec(vocab_list, input_set):
    """
    给定一个词汇集合列表和一个待分类文档, 返回词汇集合列表中各个词汇在文档中出现的次数向量
    :param vocab_list: 词汇集合列表, list形式
    :param input_set: 待分类文档, list形式
    :return: 次数向量, list
    """
    return_vector = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] += 1
    return return_vector


def split_char(input_str):
    """
    给定一个字符串, 以非数字、字母的字符为间隔进行分割, 返回分割后的列表
    :param input_str:待分割字符串, str形式
    :return:分割后列表, list
    """
    import re
    regEx = re.compile(r'\W')
    str_list = regEx.split(input_str)
    return [tok.lower() for tok in str_list if len(tok) > 2]  # 分割方式很粗暴, 之后学习正则表达式来替换


def load_sentences():
    sentences = ['This book is the best book on M.L. I have ever laid eyes upon, So I recommend this book to you.',
                 'I am your father, my son.',
                 'Your are Thor Odinson, the king of Asgard.You will be the supreme of nine sectors.',
                 'I am Iron man.',
                 'Oh, fuck, this hot dog, this, oh, shit, so delicious.Damn it.',
                 'This is my Web site\'s url: www.baidu.com or www.nchu.net']
    return sentences


def spam_test():
    doc_list, class_list, full_text = [], [], []
    for i in range(1, 26):  # 这里预先给好的数据集有 1-25 共25个, 循环读入list中
        word_list = split_char(open('email/spam/%d.txt' % i).read())  # 打开文件并且用上文的字符串分割函数分割好
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(True)
        word_list = split_char(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(False)
    # 形成了完整的数据集矩阵doc_list, full_text以及标签列表class_list
    vocab_list = create_vocab_list(doc_list)  # 用完整的数据集形成词汇集合
    training_set = list(range(50))
    # 训练集, 0-49, 数字, 这里是指在训练集中会出现的数据的下标, 即上述分割好的 25+25 个数据
    # # python3 range返回的是range对象, 不返回数组对象, 这里添加一个list函数
    test_set = []  # 测试集, 同训练集
    for i in range(10):  # 选择10个用以测试的数据
        rand_index = int(random.uniform(0, len(training_set)))  # 从 0-49 的范围中生成一个随机数
        test_set.append(training_set[rand_index])  # 在测试集中添加下标
        del(training_set[rand_index])  # 在训练集中删除下标, 保证两个集合不相交, 同时保证了两个集合的内部不重复(非set类型)
    train_mat = []
    train_classes = []
    for doc_inx in training_set:  # 建立训练集的词汇出现情况矩阵
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_inx]))  # 添加一条训练向量
        train_classes.append(class_list[doc_inx])  # 添加对应的类别
    p0, p1, p_spam = train_naive_bayes0(array(train_mat), array(train_classes))  # 计算bayes formula参数
    error_count = 0
    for doc_inx in test_set:  # 在测试集中测试
        word_vec = set_of_words2vec(vocab_list, doc_list[doc_inx])  # 用做好的词汇集合和测试集中的数据创建测试项链
        res = classify_naive_bayes0(array(word_vec), p0, p1, p_spam)
        if res != class_list[doc_inx]:  # 进行分类, 如果结果与真实情况不一致就记录
            error_count += 1
            print('The wrong data is doc_list[%d]:' % doc_inx)
            print(doc_list[doc_inx])
            print('The classifier came back with :%s, the real result is %s.' % (res, class_list[doc_inx]))
            print()
    print('The error count is :', error_count)
    print('The error rate is : ', float(error_count) / len(test_set))


def spam_test_by_bag():
    doc_list, class_list, full_text = [], [], []
    for i in range(1, 26):  # 这里预先给好的数据集有 1-25 共25个, 循环读入list中
        word_list = split_char(open('email/spam/%d.txt' % i).read())  # 打开文件并且用上文的字符串分割函数分割好
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(True)
        word_list = split_char(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(False)
    # 形成了完整的数据集矩阵doc_list, full_text以及标签列表class_list
    vocab_list = create_vocab_list(doc_list)  # 用完整的数据集形成词汇集合
    training_set = list(range(50))
    # 训练集, 0-49, 数字, 这里是指在训练集中会出现的数据的下标, 即上述分割好的 25+25 个数据
    # # python3 range返回的是range对象, 不返回数组对象, 这里添加一个list函数
    test_set = []  # 测试集, 同训练集
    for i in range(10):  # 选择10个用以测试的数据
        rand_index = int(random.uniform(0, len(training_set)))  # 从 0-49 的范围中生成一个随机数
        test_set.append(training_set[rand_index])  # 在测试集中添加下标
        del(training_set[rand_index])  # 在训练集中删除下标, 保证两个集合不相交, 同时保证了两个集合的内部不重复(非set类型)
    train_mat = []
    train_classes = []
    for doc_inx in training_set:  # 建立训练集的词汇出现情况矩阵
        train_mat.append(bag_of_words2vec(vocab_list, doc_list[doc_inx]))  # 添加一条训练向量
        train_classes.append(class_list[doc_inx])  # 添加对应的类别
    p0, p1, p_spam = train_naive_bayes0(array(train_mat), array(train_classes))  # 计算bayes formula参数
    error_count = 0
    for doc_inx in test_set:  # 在测试集中测试
        word_vec = bag_of_words2vec(vocab_list, doc_list[doc_inx])  # 用做好的词汇集合和测试集中的数据创建测试项链
        res = classify_naive_bayes0(array(word_vec), p0, p1, p_spam)
        if res != class_list[doc_inx]:  # 进行分类, 如果结果与真实情况不一致就记录
            error_count += 1
            print('The wrong data is doc_list[%d]:' % doc_inx)
            print(doc_list[doc_inx])
            print('The classifier came back with :%s, the real result is %s.' % (res, class_list[doc_inx]))
            print()
    print('The error count is :', error_count)
    print('The error rate is : ', float(error_count) / len(test_set))


def cal_most_freq(vocab_list, full_text):
    """
    给定一个词汇集合列表和一个待计数向量,
    返回以出现次数降序排序的前30个词汇及其次数,
    以(词汇, 次数)的形式组成一个list
    :param vocab_list: 词汇集合列表, list形式
    :param full_text: 待计数的向量, list形式
    :return: 词/次对列表, list
    """
    import operator as op
    freq_dict = {}
    for token in vocab_list:  # 对于词汇列表里的每一个词
        freq_dict[token] = full_text.count(token)  # 计算在待计数向量里出现的次数, 并组成词/次对
    sorted_freq = sorted(freq_dict.items(), key=op.itemgetter(1), reverse=True)  # 对词/次列表对进行降序排序
    return sorted_freq[:30]  # 返回前30个, 实际上, 应该根据需求来确定前多少个


def local_words(feed1, feed0):
    import feedparser as fp
    doc_list, class_list, full_text = [], [], []
    min_len = min(len(feed1['entries']), len(feed0['entries']))  # 得到两类数据中更少的那类的大小
    for i in range(min_len):
        word_list = split_char(feed1['entries'][i]['summary'])
        doc_list.append(word_list)  # 记录数据, 按列加入备用数据集
        full_text.extend(word_list)  # 记录数据, 全部组合成一条向量, 用以计算每个词出现的次数
        class_list.append(1)  # 记录数据, 记录每条数据的类别
        word_list = split_char(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)  # 根据数据集创建词汇集合
    top30_words = cal_most_freq(vocab_list, full_text)
    '''
    for pair in top30_words:  # 对于出现次数最多的前30个词汇
        if pair[0] in vocab_list:  # 如果在词汇列表里
            vocab_list.remove(pair[0])  # 就予以删除
    # 这里因为在英语中很多诸如that, which这样多次出现的助词, 很少的词占了很大的比例
    '''
    # 这里不知为何, 和书上的结果不一致, 添加该功能反而使得正确率大大降低
    # 根据书上的说明, 原因应该出在cal_most_freq函数里, 最后取前n位要多加考量, 后续尝试使用决策树去确定
    training_set, test_set = list(range(2*min_len)), []
    # for i in range(20):
    for i in range(int(0.25*len(training_set))):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat, train_classes = [], []
    for doc_index in training_set:
        train_mat.append(bag_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0, p1, p_spam = train_naive_bayes0(array(train_mat), array(train_classes))
    error_count = 0.0
    for doc_index in test_set:
        word_vec = bag_of_words2vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes0(array(word_vec), p0, p1, p_spam) != class_list[doc_index]:
            error_count += 1
    print('The error count is : ', error_count)
    print('The error rate is : ', float(error_count) / len(test_set) * 100, '%')
    return vocab_list, p0, p1


