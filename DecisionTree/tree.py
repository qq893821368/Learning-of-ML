from math import log
import numpy as np
import operator


def calculate_shannon_entropy(data_set):
    """
    给定一个数据集, 以最后一位为类别, 计算香农熵
    :param data_set: 数据集, list形式
    :return: 香农熵, float
    """
    """
    ┌                       ┐
        e11, e12, e13, e14
        e21, e22, e23, e24
        e31, e32, e33, e34
        e41, e42, e43, e44
    └                       ┘
    对上述矩阵, 取出对每一行的最后一位元素, 
    放入一个dict, e_name: frequency, 
    即可计算每个元素的出现概率并最后得到香农熵
    香农熵公式: -∑ Pi·log(Pi) [i=1,2,3, ... ,n]
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        label_counts[feat_vec[-1]] = label_counts.get(feat_vec[-1], 0) + 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_data_set():
    """
    创建一个用于测试的数据集
    不浮出水面是否可以生存         是否有脚蹼       属于鱼类
            是                        是              是
            是                        是              是
            是                        否              否
            否                        是              否
            否                        是              否
    对数据集进行数值化, 故是/否被化为1/0离散值
    :return: 测试数据集, list形式, 标签集, list形式
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    给定一个数据集, 指定特征轴以及特征值, 对数据集进行分割
    当特征轴上的值与给定特征值相等时, 留下该条数据
    :param data_set: 数据集, list形式
    :param axis: 特征轴, int
    :param value: 特征值, type(data_set[axis])
    :return: 分割结果集, list形式
    """
    """
        如果axis列上的值全与value相等, 则相当于从矩阵中删除某一列, 如下图
        ┌                       ┐               ┌                    ┐
            e11, e12, e13, e14                        e11, e12, e14
            e21, e22, e23, e24                        e21, e22, e24
            e31, e32, e33, e34              ->        e31, e32, e34
            e41, e42, e43, e44                        e41, e42, e44
        └                       ┘               └                    ┘
        如果axis列上的值不全与value相等, 则会剩下部分剔除了axis列的行, 如下图
        ┌                       ┐               
            e11, e12, e13, e14                    ┌                    ┐
            e21, e22, e23, e24                        e11, e12, e14
            e31, e32, e33, e34              ->        e31, e32, e34
            e41, e42, e43, e44                        e41, e42, e44
        └                       ┘               └                    ┘
        最后剩下的行, axis列的值必须与value相等
        """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    给定一个数据集, 判断用哪个特征进行分割能令信息熵最低
    对每个特征进行循环计算信息熵
    得到信息熵最低的特征即为最佳分割特征, 返回最佳分割特征的索引
    :param data_set: 数据集, list形式
    :return: 最佳分割特征索引, int
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calculate_shannon_entropy(data_set)
    best_info_gain, best_feature = 0.0, -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    """
    给定一组类别, 对其进行统计, 返回出现次数最多的类别
    :param class_list: 类别列表, list形式
    :return: 出现最多次数类别, type(class_list[0])
    """
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def prob_class(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    for key in class_count:
        class_count[key] = class_count[key] / len(class_list)
    return class_count


def create_tree(data_set, labels):
    """
    创建决策树节点
    给定数据集和标签集, 创建使得信息熵最低的决策树, 并返回根节点
    :param data_set: 数据集, list形式
    :param labels: 标签集, list形式
    :return:决策树节点, dict形式/str
    """
    """
    有时, 返回的决策树会没有用上所有的特征来分割,
    这是因为有的特征在数据中的确对数据的分类没有影响,
    这也是该算法的好处之一, 即便数据有冗余的属性,
    仍然可以准确地生成决策树, 起效的正是第一个判断-
    -语句, 如果所有数据已经是同类, 则不计其余特征
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 如果所有的数据都为同一类. 直接结束, 返回该类型
        return class_list[0]
    if len(data_set[0]) == 1:  # 如果只剩下一个特征, 返回出现最多次的类型
        return prob_class(class_list)
    # 这个条件语句是用来处理只有类别之分的数据, 此时data_set应被视为无效数据
    # 因为只有类别却没有属性值用来分割, 没有判别类型的依据
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]  # 最佳分割特征标签
    my_tree = {best_feature_label: {}}  # 创建决策树节点, 格式为{特征标签 : 归属于该类的子树}
    del(labels[best_feature])  # 删除最佳分割标签, 剩余子集好进行下一步选择新最佳分割标签 (这一步会对原list进行操作导致原list变小)
    # del不是直接从内存中删除, 而是删除这个指针引用, 此处等效于labels.remove(best_feature_label)
    feature_values = [example[best_feature] for example in data_set]  # 把每条数据的最佳分割特征值取出组成一个列表
    unique_vals = set(feature_values)  # 最佳分割特征值集合
    for value in unique_vals:  # 在集合里, 决策树节点[最佳特征标签][最佳分割特征值] = 子决策树节点
        sub_labels = labels[:]  # 因为每一个决策树节点之后, 可能是直接判别类别, 也可能是继续判断的一个子节点
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
        # 所以, 返回的决策树节点可以是dict也可以是str, 那么就调用本方法继续生成下一个节点
        # 递归地调用本方法, 通过返回新的节点给上一层的父节点, 最终生成一个根节点作为决策树的入口节点
        # # 这里注意, 每个节点下, 对best_feature_label这个key, 每种value应该都有一个子节点, 由本方法生成
        # # 所以, 这里是一个循环, 对最佳分割特征中每一个独特的特征值都要调用一次本方法,
        # # 并且是用以该value分割好的子data_set去新生成一个子节点
        # # 因为这里根节点已经是对某一feature来进行分割, 在这一feature下的每个feature值, 都对应了一个子data_set
        # # 对于这个子data_set还需要用同样的方法进行分割, 形成一颗完成的子树作为子节点嵌入根节点
    return my_tree


def classify(input_tree, feature_labels, test_vector):
    first_string = list(input_tree.keys())[0]
    second_dict = input_tree[first_string]
    feature_index = feature_labels.index(first_string)
    class_label = None
    for key in second_dict.keys():
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'wb')
    pickle.dump(str(input_tree), fw)
    fw.close()


def grab_tree(file_name):
    import pickle
    fr = open(file_name, 'rb')
    return pickle.load(fr)


