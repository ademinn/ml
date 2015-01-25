import os
import sys
import math
import random
from collections import defaultdict


class Message(object):
    def __init__(self, subject, text, is_spam, msg_name=None):
        self.subject = subject
        self.text = text
        self.is_spam = is_spam
        self.msg_name = msg_name


class MessageLoader(object):
    def __init__(self, data_dir, train, valid):
        self.data_dir = os.path.abspath(data_dir)
        parts = [os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir) if 'part' in d]
        random.shuffle(parts)
        parts_len = len(parts)
        train_sep, valid_sep = (int(parts_len * x) for x in (train, train + valid))
        train_parts = parts[:train_sep]
        valid_parts = parts[train_sep:valid_sep]
        test_parts = parts[valid_sep:]
        self.train = self.__load_parts(train_parts)
        self.valid = self.__load_parts(valid_parts)
        self.test = self.__load_parts(test_parts)

    @staticmethod
    def __load_parts(parts):
        result = list()
        for part in parts:
            result.extend(MessageLoader.__load_part(part))
        return result

    @staticmethod
    def __load_part(part):
        msg_files = [os.path.join(part, f) for f in os.listdir(part)]
        return [MessageLoader.__parse_message(f) for f in msg_files]

    @staticmethod
    def __parse_message(msg_file):
        msg_name = os.path.basename(msg_file)
        is_spam = 'spmsg' in msg_name
        with open(msg_file) as f:
            subj_str = f.readline()
            subject = map(int, subj_str.split()[1:])
            f.readline()
            text = map(int, f.read().split()[1:])
            return Message(subject, text, is_spam, msg_name)


class WordCounter(object):
    def __init__(self):
        self.__words = dict()
        self.__total = 0

    def add_words(self, words):
        self.__total += len(words)
        for w in words:
            if w not in self.__words:
                self.__words[w] = 0
                self.__total += 1
            self.__words[w] += 1

    def calc_prob(self, w):
        w_cnt = 1 if w not in self.__words else self.__words[w]
        return 1.0 * w_cnt / self.__total


class MessageCounter(object):
    def __init__(self):
        self.subject = WordCounter()
        self.text = WordCounter()
        self.total = 0

    def add_message(self, msg):
        self.total += 1
        self.subject.add_words(msg.subject)
        self.text.add_words(msg.text)


class BayesClassifier(object):
    def __init__(self, msgs):
        self.legit = MessageCounter()
        self.spam = MessageCounter()
        self.threshold = 0.0

        for msg in msgs:
            self.add_message(msg)

    def add_message(self, msg):
        if msg.is_spam:
            self.spam.add_message(msg)
        else:
            self.legit.add_message(msg)

    def calc_log_likehood(self, msg):
        likehood = 1.0 * self.spam.total / self.legit.total
        for w in msg.subject:
            likehood += math.log(self.spam.subject.calc_prob(w) / self.legit.subject.calc_prob(w))
        for w in msg.text:
            likehood += math.log(self.spam.text.calc_prob(w) / self.legit.text.calc_prob(w))
        return likehood

    def find_threshold(self, valid):
        likehoods = map(self.calc_log_likehood, valid)
        res = zip(valid, likehoods)
        res.sort(lambda x, y: cmp(y[1], x[1]))
        self.threshold = float('inf')
        ok_count = len([v for v in valid if not v.is_spam])
        if ok_count == 0:  # only spam
            self.threshold = float('-inf')
        else:
            max_ok_count = ok_count
            for i in range(len(res) - 1):
                if res[i][0].is_spam:
                    ok_count += 1
                else:
                    ok_count -= 1
                if ok_count > max_ok_count:
                    max_ok_count = ok_count
                    self.threshold = (res[i][1] + res[i + 1][1]) / 2

    def is_spam_verbose(self, msg):
        likehood = self.calc_log_likehood(msg)
        return self.calc_log_likehood(msg) > self.threshold, likehood

    def is_spam(self, msg):
        return self.is_spam_verbose(msg)[0]

    def test_msg(self, msg):
        is_spam, likehood = self.is_spam_verbose(msg)
        res = is_spam == msg.is_spam
        str_res = 'ok' if res else 'fail'
        print('%20s    %10.2f    %s' % (msg.msg_name, likehood, str_res))
        return res

    def test(self, msgs):
        results = map(self.test_msg, msgs)
        ok_count = len([r for r in results if r])
        print('threshold = %.2f' % self.threshold)
        print('ok %d from %d; %.2f%%' % (ok_count, len(msgs), 100.0 * ok_count / len(msgs)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('data dir not specified')
        sys.exit(1)
    loader = MessageLoader(sys.argv[1], 0.6, 0.2)
    classifier = BayesClassifier(loader.train)
    classifier.find_threshold(loader.valid)
    classifier.test(loader.test)
