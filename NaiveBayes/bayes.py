from numpy import *


def load_data_set():
    """
    创建一个默认的数据集
    :return: 文本列表, list
    :return: 词条向量, list, 表示所给的文本列表的对应项是否出现侮辱性词语
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage', ],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buting', 'worthless', 'dog', 'food', 'stupid']]
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
    :param train_matrix: 文档的词汇出现情况矩阵, 描述每个文档中出现词汇集合中的词汇的情况
    :param train_category: 文档类别列表, 描述每个文档是否侮辱性文档
    :return:非侮辱性文档条件下各词汇出现概率, numpy.array
    :return:侮辱性文档条件下各词汇出现概率, numpy.array
    :return:侮辱性文档概率, float
    """
    num_train_docs = len(train_matrix)  # 文档个数
    num_words = len(train_matrix[0])  # 词汇个数
    p_abusive = sum(train_category) / float(num_train_docs)  # 侮辱性文档概率
    p0_denom = 0.0  # p0分母
    p1_denom = 0.0  # p1分母
    p0_num = zeros(num_words)  # p(词汇i出现|非侮辱性文档)
    p1_num = zeros(num_words)  # p(词汇i出现|侮辱性文档)
    for i in range(num_train_docs):  # 每个文档
        if train_category[i] == 1:  # 如果文档是侮辱性文档
            p1_num += train_matrix[i]  # 在侮辱性文档的词汇概率矩阵中记录
            p1_denom += sum(train_matrix[i])  # 记录词汇集合中的所有词汇的出现总次数, 上句为分子, 本句为分母, 结尾相除可的概率
        else:  # 如果不是侮辱性文档
            p0_num += train_matrix[i]  # 在非侮辱性文档的词汇概率矩阵中记录
            p0_denom += sum(train_matrix[i])  # 同上, 记录
    if p0_denom == 0.0:     # -------------------------------------------------------------------
        p0_denom = 1.0      # 这一段自己添加, 可以处理词汇集合中只有侮辱性词语, 诸如"傻逼"、"fuck"
    if p1_denom == 0.0:     # 这样公认的侮辱性词汇的情况, 即文档可能不出现词汇集合中的词语的情况
        p1_denom = 1.0      # -------------------------------------------------------------------
    p1_vector = p1_num / p1_denom  # 相除求得侮辱性文档的词汇出现次数/词汇集合中的词汇出现的总次数, 即在侮辱性文档的词汇概率矩阵中记录
    p0_vector = p0_num / p0_denom  # 同上, 求得在非侮辱性文档的词汇概率矩阵中记录

    return p0_vector, p1_vector, p_abusive  # 返回(非)侮辱性文档条件下各词汇出现概率, (非)侮辱性文档的概率

