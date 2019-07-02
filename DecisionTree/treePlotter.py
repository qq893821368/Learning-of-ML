# -- coding:utf-8 --
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_text, center_point, parent_point, node_type):
    create_plot.axl.annotate(node_text, xy=parent_point, xycoords='axes fraction', xytext=center_point, ha='center',
                            bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_leafs_num(my_tree):
    """
    给定一个决策树, 计算总叶节点数
    :param my_tree: 决策树根节点
    :return: leafs_num, int
    """
    leafs_num = 0
    """
    原语句 : first_str = my_tree.keys()[0]
    这是Python2.7的语法, 我用的是Python3.6, 原解决方案如下
    temp = my_tree.keys()
    keys = []
    for key in temp:
        keys.append(key)
    first_str = keys[0]
    """
    # 更简洁的解决方案, 直接转换成list
    first_str = list(my_tree.keys())[0]  # 获取第一个key
    second_dict = my_tree[first_str]  # 获取和第一个key对应的值, 可能是另一个dict
    for key in second_dict.keys():  # 在获取到的新dict里去查看每一个key所对应的值是不是也是dict
        if type(second_dict[key]).__name__ == 'dict':
            leafs_num += get_leafs_num(second_dict[key])  # 如果对应的value也是dict, 就计算这个dict的叶节点数
        else:
            leafs_num += 1  # 如果对应的value不是dict, 那么说明到达叶节点, 总数+1
    return leafs_num


def get_tree_depth(my_tree):
    """
    给定一个决策树, 计算其深度
    :param my_tree: 决策树根节点
    :return: depth, int
    """
    depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = get_tree_depth(second_dict[key]) + 1  # 这里不能直接return get_tree_depth(second_dict[key]) + 1
        else:
            this_depth = 1  # 同样的这里也不能直接return 1, 因为要对整个子dict进行子树的求深度
        if this_depth > depth:  # 如果直接return, 会在遇到第一个非dict的子节点时便结束递归, 导致深度计算错误
            depth = this_depth
    return depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                     {'no surfacing': {1: {'fippers': {0: 'no', 1: 'yes'}}, 0: 'no'}}]
    return list_of_trees[i]


def plot_mid_text(center_point, parent_point, txt_str):
    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    create_plot.axl.text(x_mid, y_mid, txt_str)


def plot_tree(my_tree, parent_point, node_txt):
    leafs_num = get_leafs_num(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    center_point = (plot_tree.x_off + (1.0 + float(leafs_num)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(center_point, parent_point, node_txt)
    plot_node(first_str, center_point, parent_point, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0/plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], center_point, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), center_point, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), center_point, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')  # 编号为1, 背景色白色
    # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
    # num编号, figsize宽高, dpi分辨率(默认80), facecolor背景色, edgecolor边框色, frameon是否显示边框
    fig.clf()  # clear figure 清楚整个当前数字
    axprops = dict(xticks=[], yticks=[])  # 键值对
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)  # 第1行第1列第1幅图, 不显示边框, 把上面的键值对做参数
    # subplot(nrows,ncols,plotnum,sharex,sharey,subplot_kw,**fig_kw)
    # 参数111表示nrows=1, ncols=1, plotnum=1, 也可以用"1, 1, 1"表示
    # sharex分享x轴定义即各子图的x轴有一致的规格, sharey同sharex, subplot_kw把字典的关键字传递给add_subplot()来创建每个子图
    plot_tree.total_w = float(get_leafs_num(in_tree))  # 宽度为叶子节点数
    plot_tree.total_d = float(get_tree_depth(in_tree))  # 深度为树的深度
    plot_tree.x_off = -0.5 / plot_tree.total_w  # x坐标, -(0.5/宽度)
    plot_tree.y_off = 1.0  # y坐标, 1.0
    plot_tree(in_tree, (0.5, 1.0), '')  # 用传入的决策树, 以(0.5, 1.0)为起始点坐标, 根节点文本为空, 绘制决策树图形
    plt.show()


