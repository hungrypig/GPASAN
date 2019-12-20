import copy
import pickle
import random
import sys
from collections import defaultdict

import numpy as np


def time_data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    Time = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    time_train = {}
    time_valid = {}
    time_test = {}
    # assume user/item index starting from 1
    f = open('data/%s_time_4.txt' % fname, 'r')
    for line in f:
        u, i, _, t = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        t = int(t)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        Time[u].append(t)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            time_train[user] = Time[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            time_train[user] = Time[user][:-2]
            user_valid[user] = []
            time_valid[user] = []
            user_valid[user].append(User[user][-2])
            time_valid[user].append(Time[user][-2])
            user_test[user] = []
            time_test[user] = []
            user_test[user].append(User[user][-1])
            time_test[user].append(Time[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum, time_train, time_valid, time_test]


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s_time_4.txt' % fname, 'r')
    for line in f:
        u, i, _, _d = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def sub_data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        user_sub = 1

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # user_train[user] = User[user]
            # user_valid[user] = []
            # user_test[user] = []
            break
        else:
            for seqlen in range(3, nfeedback + 1):
                user_train[user_sub] = User[user][:seqlen - 2]
                user_valid[user_sub] = []
                user_valid[user_sub].append(User[user][seqlen - 2])
                user_test[user_sub] = []
                user_test[user_sub].append(User[user][seqlen - 1])
                user_sub += 1

    pickle.dump(user_train, open(fname + '_data/train.txt', 'wb'))
    pickle.dump(user_valid, open(fname + '_data/valid.txt', 'wb'))
    pickle.dump(user_test, open(fname + '_data/test.txt', 'wb'))
    pickle.dump(user_sub - 1, open(fname + '_data/usernum.txt', 'wb'))
    pickle.dump(itemnum, open(fname + '_data/itemnum.txt', 'wb'))

    return [user_train, user_valid, user_test, user_sub - 1, itemnum]


def evaluate_time(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum, time_train, time_valid, time_test] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0

    NDCG_5 = 0.0
    HT_5 = 0.0

    # NDCG_15 = 0.0
    # HT_15 = 0.0

    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_pad = np.zeros([args.maxlen], dtype=np.float32)
        time_interval = np.zeros((args.maxlen, args.maxlen), dtype=np.float32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        time_pad[idx] = time_valid[u][0]
        j = len(time_train[u]) - 1
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            time_pad[idx] = time_train[u][j]
            j -= 1
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)

        start_idx = args.maxlen - min(len(train[u]) + 1, args.maxlen)
        for i in range(start_idx, args.maxlen):
            for j in range(i, args.maxlen):
                time_interval[i][j] = time_pad[j] - time_pad[i]

        item_idx = [test[u][0]]
        for _ in range(args.candidate_count - 1):  # 一个正例，99个随机负例用作评测
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # item_idx = range(1, itemnum + 1)
        # =======================================repeat==================================
        viewed_item = dataset[0][u]
        # viewed_item = train[u]
        predictions_explore = -model.predict(sess, [u], [seq], item_idx, [time_interval])[0][0]
        predictions_repeat = -model.predict(sess, [u], [seq], item_idx, [time_interval])[1][0]

        for viewed_index in viewed_item:
            if viewed_index in item_idx:
                idx = item_idx.index(viewed_index)
                predictions_explore[idx] = predictions_repeat[idx]
        predictions = predictions_explore

        # predictions = -model.predict(sess, [u], [seq], item_idx, [time_interval])
        # predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]  # ground truth 在推荐中的排序
        # ranked_index = predictions.argsort()
        # rank = ranked_index.tolist().index(test[u][0] - 1)

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

        # if rank < 15:
        #     NDCG_15 += 1 / np.log2(rank + 2)
        #     HT_15 += 1
        # if valid_user % 100 == 0:
        #     print('.', end=' ')
        #     sys.stdout.flush()

    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user

# def evaluate(model, dataset, args, sess):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#
#     NDCG_5 = 0.0
#     HT_5 = 0.0
#     NDCG_20 = 0.0
#     HT_20 = 0.0
#     valid_user = 0.0
#
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#
#         if len(train[u]) < 1 or len(test[u]) < 1:
#             continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1:
#                 break
#         rated = set(train[u])
#         rated.add(0)
#
#         item_idx = [test[u][0]]
#         for _ in range(100):  # 一个正例，99个随机负例用作评测
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated:
#                 t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#
#         # item_idx = range(1, itemnum + 1)
#         # =======================================repeat==================================
#         viewed_item = dataset[0][u]
#         predictions_explore = -model.predict(sess, [u], [seq], item_idx)[0][0]
#         predictions_repeat = -model.predict(sess, [u], [seq], item_idx)[1][0]
#
#         for viewed_index in viewed_item:
#             if viewed_index in item_idx:
#                 idx = item_idx.index(viewed_index)
#                 predictions_explore[idx] = predictions_repeat[idx]
#         predictions = predictions_explore
#
#         # predictions = -model.predict(sess, [u], [seq], item_idx)
#         # predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0]  # ground truth 在推荐中的排序
#
#         # ranked_index = predictions.argsort()
#         # rank = ranked_index.tolist().index(test[u][0] - 1)
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG_20 += 1 / np.log2(rank + 2)
#             HT_20 += 1
#         if valid_user % 100 == 0:
#             print('.', end=' ')
#             sys.stdout.flush()
#
#         if rank < 5:
#             NDCG_5 += 1 / np.log2(rank + 2)
#             HT_5 += 1
#         if valid_user % 100 == 0:
#             print('.', end=' ')
#             sys.stdout.flush()
#
#     return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user


# def evaluate_valid(model, dataset, args, sess):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1:
#                 break
#
#         rated = set(train[u])
#         rated.add(0)
#
#         # item_idx = range(itemnum)
#         # item_idx = [valid[u][0]]
#         # for _ in range(100):
#         #     t = np.random.randint(1, itemnum + 1)
#         #     while t in rated:
#         #         t = np.random.randint(1, itemnum + 1)
#         #     item_idx.append(t)
#
#         item_idx = range(1, itemnum + 1)
#         # ====================================repeat========================================
#         # viewed_item = dataset[0][u]
#         # predictions_explore = -model.predict(sess, [u], [seq], item_idx)[0][0]
#         # predictions_repeat = -model.predict(sess, [u], [seq], item_idx)[1][0]
#         #
#         # for viewed_index in viewed_item:
#         #     predictions_explore[viewed_index - 1] = predictions_repeat[viewed_index - 1]
#         # predictions = predictions_explore
#
#         predictions = -model.predict(sess, [u], [seq], item_idx)
#         predictions = predictions[0]
#
#         # rank = predictions.argsort().argsort()[0]
#         ranked_index = predictions.argsort()
#         rank = ranked_index.tolist().index(valid[u][0] - 1)  # 因为range 从1 到 itemnum，所以预测返回列表下标从0 到 itemnum-1，需要错位相对。
#
#         valid_user += 1
#
#         if rank < 20:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end=' ')
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user
