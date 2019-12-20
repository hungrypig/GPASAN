from multiprocessing import Process, Queue

import numpy as np


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, time):  # 创造训练数据,添加时间
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)  # 创建输入序列，不足补0
        time_pad = np.zeros([maxlen], dtype=np.float32)  # 前面用0 和seq 补齐
        last_time_idx = len(time[user]) - 2
        pos = np.zeros([maxlen], dtype=np.int32)
        pos_mask = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]  # 最后一个item
        idx = maxlen - 1
        time_interval = np.zeros((maxlen, maxlen), dtype=np.float32)

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):  # 除去最后一个，然后翻转，一个用户
            seq[idx] = i  # 从后往前，写index，前面补0
            time_pad[idx] = time[user][last_time_idx]  # 时间也是从后往前写
            last_time_idx -= 1
            pos[idx] = nxt  # 放一个用户序列最后一个值 ground——truth。
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)  # 负样例随机取一个
            nxt = i  # 正样例为从后往前，一个个，一个用户序列，多次划分
            idx -= 1
            if idx == -1:
                break

        start_idx = maxlen - min(len(user_train[user]), maxlen)
        for i in range(start_idx, maxlen):
            for j in range(i, maxlen):
                time_interval[i][j] = time_pad[j] - time_pad[i]

        for j in range(len(pos)):
            pos_mask[j] = pos[j] in seq[:j + 1]

        return (user, seq, pos, neg, time_interval, pos_mask)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(list(zip(*one_batch)))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, time_train=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      time_train
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
