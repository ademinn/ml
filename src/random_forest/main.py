import os
import sys
import random
import math

from collections import Counter

FEATURES = 10000


class Node:
    def __init__(self, feature, value, left, right):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        def tab(s):
            return '\n'.join(['    ' + new_s for new_s in s.splitlines()])
        left_str = tab(str(self.left))
        right_str = tab(str(self.right))

        return "feature #{} < {} -> \n{}\n{}\n".format(self.feature, self.value, left_str, right_str)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Node) and \
            self.feature == other.feature and \
            self.value == other.value and \
            self.left == other.left and \
            self.right == other.right

class Leaf:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return "leaf {}".format(self.label)
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Leaf) and \
            self.label == other.label

class Tree:
    def __init__(self, root):
        self.root = root

    def classify(self, x):
        cur = self.root
        while not isinstance(cur, Leaf):
            if x[cur.feature] < cur.value:
                cur = cur.left
            else:
                cur = cur.right
        return cur.label

    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Tree) and \
            self.root == other.root


def train(data):
    return Tree(split(data))


def split(data):
    best_gini = float('+inf')
    best_feature = -1
    best_left = None
    best_right = None
    for _ in range(int(math.sqrt(FEATURES))):
        feature = random.randint(0, FEATURES - 1)
        value, left, right = data.split(feature)
        cur_gini = gini(left, right)
        if cur_gini < best_gini:
            best_gini = cur_gini
            best_feature = feature
            best_left = left
            best_right = right
            best_value = value

    if best_left.data_len == 0 or best_right.data_len == 0:
        return Leaf(data.dominant_label)

    if min(best_left.labels) == max(best_left.labels):
        left = Leaf(best_left.labels[0])
    else:
        left = split(best_left)

    if min(best_right.labels) == max(best_right.labels):
        right = Leaf(best_right.labels[0])
    else:
        right = split(best_right)

    return Node(best_feature, best_value, left, right)


class TrainData(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        self.data_len = len(self.data)
        if self.data_len:
            self.dominant_label = Counter(self.labels).most_common(1)[0][0]
        freqs = Counter(self.labels)
        self.gini_coef = (1 - sum((1.0 * f / self.data_len) ** 2 for f in freqs.values())) * self.data_len

    def split(self, feature):
        values = [self.data[i][feature] for i in range(self.data_len)]
        srt = sorted(values)
        if self.data_len % 2 == 1:
            median = srt[(self.data_len - 1) / 2]
        else:
            i = self.data_len / 2 - 1
            median = (srt[i] + srt[i + 1]) / 2.0
        lesser = list()
        greater = list()
        for i in range(self.data_len):
            if values[i] < median:
                lesser.append((self.data[i], self.labels[i]))
            else:
                greater.append((self.data[i], self.labels[i]))
        if lesser:
            lesser_data, lesser_labels = zip(*lesser)
        else:
            lesser_data = list()
            lesser_labels = list()
        
        if greater:
            greater_data, greater_labels = zip(*greater)
        else:
            greater_data = list()
            greater_labels = list()
        return median, TrainData(lesser_data, lesser_labels), TrainData(greater_data, greater_labels)


def gini(left, right):
    return 1.0 * (left.gini_coef + right.gini_coef) / (left.data_len + right.data_len)


def read_data(data_file):
    with open(data_file) as f:
        return [map(int, l.split()) for l in f.readlines()]


def read_labels(labels_file):
    with open(labels_file) as f:
        return map(int, f.readlines())


if __name__ == '__main__':
    data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        trees_count = int(sys.argv[2])
    else:
        trees_count = 100

    def file_path(file_name):
        return os.path.join(data_dir, file_name)

    train_data = read_data(file_path('arcene_train.data'))
    train_labels = read_labels(file_path('arcene_train.labels'))
    tr = TrainData(train_data, train_labels)

    trees = list()
    for _ in range(trees_count):
        tree = train(tr)
        trees.append(tree)

    test_data = read_data(file_path('arcene_valid.data'))
    test_labels = read_labels(file_path('arcene_valid.labels'))

    for data, label in zip(test_data, test_labels):
        res_labels = [t.classify(data) for t in trees]
        best = Counter(res_labels).most_common(1)[0][0]
        if best == label:
            print('ok')
        else:
            print('fail')

