import re


class Member:

    def __init__(self):
        self.score_map = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.name = input('Input your mark:')

    def assess(self):
        self.score_map['A'] = int(input('Input A\'s score:'))
        self.score_map['B'] = int(input('Input B\'s score:'))
        self.score_map['C'] = int(input('Input C\'s score:'))
        self.score_map['D'] = int(input('Input D\'s score:'))

    def mat2map(self):
        mat = input('---%s $ Input A、B、C、D \'s scores:' % self.name)
        mat = re.split(r" +", mat)
        self.score_map['A'] = int(mat[0])
        self.score_map['B'] = int(mat[1])
        self.score_map['C'] = int(mat[2])
        self.score_map['D'] = int(mat[3])


print('-----Create 4 member:')
A, B, C, D = Member(), Member(), Member(), Member()
A.mat2map()
B.mat2map()
C.mat2map()
D.mat2map()

score = []
score.append(A.score_map['A']*0.1 + (B.score_map['A']+C.score_map['A']+D.score_map['A'])*0.3)
score.append(B.score_map['B']*0.1 + (A.score_map['B']+C.score_map['B']+D.score_map['B'])*0.3)
score.append(C.score_map['C']*0.1 + (B.score_map['C']+A.score_map['C']+D.score_map['C'])*0.3)
score.append(D.score_map['D']*0.1 + (B.score_map['D']+C.score_map['D']+A.score_map['D'])*0.3)
sum = 0.0
for each in score:
    sum += each
retain = A.score_map['A']+A.score_map['B']+A.score_map['C']+A.score_map['D'] - sum
for i in range(len(score)):
    score[i] += (retain - retain % 4) / 4
retain = retain % 4
mi = 100
index = 0
for i in range(len(score)):
    if mi > score[i]:
        mi = score[i]
        index = i
score[index] += retain
print('---Administrator $ Score:[A:%d, B:%d, C:%d, D:%d]' % (score[0], score[1], score[2], score[3]))
