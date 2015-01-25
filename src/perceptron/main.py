import os
import sys
import struct
import argparse


LABELS_NUM = 10


class ByteReader(object):
    def __init__(self, data):
        self.__data = data

    def __next_bytes(self, num):
        res = self.__data[:num]
        self.__data = self.__data[num:]
        return res

    def read_int(self):
        raw_res = self.__next_bytes(4)
        return struct.unpack('>i', raw_res)[0]

    def read_bytes(self, num):
        return bytearray(self.__next_bytes(num))


def read_labels(file_name, total=-1):
    with open(file_name) as f:
        reader = ByteReader(f.read())
    reader.read_int()
    cnt = reader.read_int()
    if total < 0:
        total = cnt
    return reader.read_bytes(total)


def read_images(file_name, total=-1):
    with open(file_name) as f:
        reader = ByteReader(f.read())
    reader.read_int()
    cnt = reader.read_int()
    if total < 0:
        total = cnt
    rows = reader.read_int()
    cols = reader.read_int()
    img_size = rows * cols
    raw_data = reader.read_bytes(cnt * img_size)
    result = list()
    for i in range(total):
        if i % 1000 == 0:
            print('read %d images' % i)
        image = raw_data[:img_size]
        raw_data = raw_data[img_size:]
        result.append(image)
    return img_size, result


def build_images(images_file, labels_file, total=-1):
    img_size, images = read_images(images_file, total)
    labels = read_labels(labels_file, total)
    return img_size, map(Image, images, labels)


def normalize(color):
    return (color - 128.0) / 128


class Image(object):
    def __init__(self, data, label):
        self.data = map(normalize, data)
        self.label = label


class Classifier(object):
    def __init__(self, img_size, label, data=None):
        self.label = label
        self.img_size = img_size
        if data is None:
            self.data = [0.0] * img_size
        else:
            self.data = data

    def calc(self, data):
        return sum([x * y for x, y in zip(self.data, data)])

    def add(self, data):
        for i in range(self.img_size):
            self.data[i] += data[i]

    def sub(self, data):
        for i in range(self.img_size):
            self.data[i] -= data[i]


class Perceptron(object):
    def __init__(self):
        self.img_size = 0
        self.classifiers = list()

    def init(self, img_size):
        self.img_size = img_size
        self.classifiers = [Classifier(self.img_size, label) for label in range(LABELS_NUM)]

    def train(self, images):
        cnt = 0
        for image in images:
            if cnt % 1000 == 0:
                print('%d done' % (100 * cnt / img_size))
            cnt += 1
            cfr = self.max_classifier(image.data)
            if cfr.label != image.label:
                cfr.sub(image.data)
                self.classifiers[image.label].add(image.data)

    def max_classifier(self, data):
        max_res = float('-inf')
        max_cfr = None
        for cfr in self.classifiers:
            res = cfr.calc(data)
            if res > max_res:
                max_res = res
                max_cfr = cfr
        return max_cfr

    def test(self, images):
        ok_count = 0
        for image in images:
            cfr = self.max_classifier(image.data)
            if cfr.label != image.label:
                print('%d != %d' % (cfr.label, image.label))
            else:
                ok_count += 1
        total = len(images)
        print('ok: %d, total: %d; %.2f%%' % (ok_count, total, 100.0 * ok_count / total))

    def dump(self, dump_file):
        with open(dump_file, 'w') as f:
            f.write('%d %d\n' % (self.img_size, len(self.classifiers)))
            for cfr in self.classifiers:
                f.write('%d\n' % cfr.label)
                for v in cfr.data:
                    f.write('%f ' % v)
                f.write('\n')

    def read(self, dump_file):
        with open(dump_file) as f:
            self.img_size, cnt = map(int, f.readline().split())
            self.classifiers = list()
            for i in range(cnt):
                label = int(f.readline())
                data = map(float, f.readline().split())
                self.classifiers.append(Classifier(self.img_size, label, data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir')
    parser.add_argument('--dump-path', dest='dump_path')
    parser.add_argument('--read-path', dest='read_path')
    parser.add_argument('--total-train', dest='total_train', type=int, default=-1)
    parser.add_argument('--total-test', dest='total_test', type=int, default=-1)
    parser.add_argument('--train-iters', dest='train_iters', type=int, default=10)
    args = parser.parse_args()

    def file_path(file_name):
        return os.path.join(args.data_dir, file_name)

    p = Perceptron()

    if args.read_path:
        p.read(args.read_path)
    else:
        img_size, train = build_images(file_path('train-images-idx3-ubyte'), file_path('train-labels-idx1-ubyte'), args.total_train)
        p.init(img_size)
        for i in range(args.train_iters):
            p.train(train)
    
    if args.dump_path:
        p.dump(args.dump_path)

    test = build_images(file_path('t10k-images-idx3-ubyte'), file_path('t10k-labels-idx1-ubyte'), args.total_test)[1]
    p.test(test)
