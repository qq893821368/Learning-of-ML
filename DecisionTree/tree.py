from math import log
import numpy as np
import operator


def calculate_shannon_entropy(data_set):
    """
    给定一个数据集, 以最后一位为类别, 计算香农熵
    :param data_set: 数据集, list形式
    :return: 香农熵, float
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
    得到信息熵最低的特征即为最佳分割特征
    :param data_set: 数据集, list形式
    :return: 最佳分割特征, int
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


def create_tree(data_set, labels):
    """
    创建决策树节点
    给定数据集和标签集, 创建使得信息熵最低的决策树, 并返回根节点
    :param data_set: 数据集, list形式
    :param labels: 标签集, list形式
    :return:决策树节点, dict形式/str
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 如果所有的数据都为同一类. 直接结束, 返回该类型
        return class_list[0]
    if len(data_set[0]) == 1:  # 如果只剩下一个特征, 返回出现最多次的类型
        return majority_count(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]  # 最佳分割特征标签
    my_tree = {best_feature_label: {}}  # 创建决策树节点, 格式为{特征标签 : 归属于该类的子树}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]  # 把每条数据的最佳分割特征值取出
    unique_vals = set(feature_values)  # 最佳分割特征值集合
    for value in unique_vals:  # 在集合里, 决策树节点[最佳特征标签][最佳分割特征值] = 子决策树节点
        sub_labels = labels[:]  # 因为每一个决策树节点之后, 可能是直接判别类别, 也可能是继续判断的一个子节点
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
        # 所以, 返回的决策树节点可以是dict也可以是str, 那么就调用本方法继续生成下一个节点
        # 递归地调用本方法, 通过返回新的节点给上一层的父节点, 最终生成一个根节点作为决策树的入口节点
    return my_tree



