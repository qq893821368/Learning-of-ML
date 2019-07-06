import re
import string
import NaiveBayes.bayes as bayes


def split_char(input_str):
    """
    给定一个字符串, 以非数字、字母的字符为间隔进行分割, 返回分割后的列表
    :param input_str:待分割字符串, str形式
    :return:分割后列表, list
    """
    import re
    input_str = input_str.split()
    regEx = re.compile(r'\W')
    return_list = []
    for word in input_str:
        pass
    return return_list# [tok.lower() for tok in str_list if len(tok) > 2]  # 分割方式很粗暴, 之后学习正则表达式来替换


reg = re.compile(r'www[.]\S*[.com]')
sentence = bayes.load_sentences()[-1]
words = sentence.split()
for w in words:
    print([w], end='')
print()
word = []
for w in words:
    word.append(w.strip(string.punctuation))
words = []
for w in word:
    words.extend(item.group().split('.')[1] for item in re.finditer(r'www[.]\S*.com|www[.]\S*.net', w))

for w in words:
    print(w, end=' ')
