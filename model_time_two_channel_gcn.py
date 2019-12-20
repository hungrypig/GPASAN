from tensorflow.python.ops import math_ops

from modules_time import *


class Model():
    def __init__(self, usernum, itemnum, args, norm_adj, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.time_interval = tf.placeholder(tf.float32, shape=(None, args.maxlen, args.maxlen))
        self.pos_mask = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        self.norm_adj = norm_adj
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        # ------------------------------------periodic---------------------------------
        cosin = math_ops.cos
        self.k_periodic_table = tf.get_variable('k_periodic', [itemnum, 1], dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer())
        self.k_periodic_items = tf.nn.embedding_lookup(self.k_periodic_table, self.input_seq)
        self.time_periodic_intervals = (cosin(self.time_interval * self.k_periodic_items) + 1) / 2

        # ------------------------------------decaying----------------------------------------
        self.k_son_table = tf.get_variable('k_son', [itemnum, 1], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer())
        self.k_son_items = tf.nn.embedding_lookup(self.k_son_table, self.input_seq)
        self.time_decading_intervals = tf.tanh(1 / self.k_son_items * (self.time_interval + 0.0001))

        # -----------------------------------------gcn weights------------------------------------
        self.n_layers = args.n_layers
        initializer = tf.contrib.layers.xavier_initializer()
        # multi layers
        all_weights = dict()
        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([args.hidden_units, args.hidden_units]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, args.hidden_units]), name='b_gc_%d' % k)

        # ---------------------------------------w_re----------------------------------------------
        self.w_re = tf.get_variable('W_re', shape=[args.hidden_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(0, 1))

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )  # 返回 seq 的 embedding矩阵和 所有item 矩阵

            self.norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            adj_multi_embedding = tf.sparse_tensor_dense_matmul(self.norm_adj, item_emb_table)
            for k in range(self.n_layers):
                new_item_embeddings = tf.nn.leaky_relu(
                    tf.matmul(adj_multi_embedding, all_weights['W_gc_%d' % k]) + all_weights['b_gc_%d' % k])
                new_item_embeddings = tf.layers.dropout(new_item_embeddings,
                                                        rate=args.dropout_rate,
                                                        training=tf.convert_to_tensor(self.is_training))
                self.seq = tf.nn.embedding_lookup(new_item_embeddings, self.input_seq)

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention_periodic",
                                                   time_interval=self.time_decading_intervals)

                    self.repeat_seq = multihead_attention(queries=normalize(self.seq),
                                                          keys=self.seq,
                                                          num_units=args.hidden_units,
                                                          num_heads=args.num_heads,
                                                          dropout_rate=args.dropout_rate,
                                                          is_training=self.is_training,
                                                          causality=True,
                                                          scope="self_attention_decaying",
                                                          time_interval=self.time_periodic_intervals)

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

                    self.repeat_seq = feedforward(normalize(self.repeat_seq),
                                                  num_units=[args.hidden_units, args.hidden_units],
                                                  dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.repeat_seq *= mask

            self.seq = normalize(self.w_re * self.seq)
            self.repeat_seq = normalize((1 - self.w_re) * self.repeat_seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])  # 降成一维列表。
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq,
                             [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])  # 最后的session表示，需要优化, 二维
        repeat_seq_emb = tf.reshape(self.repeat_seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        #  测试集的item——embedding和得分
        self.test_item = tf.placeholder(tf.int32, shape=args.candidate_count)
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))  # 二维， 【batchsize*maxlen， 101】
        self.test_logits = tf.reshape(self.test_logits,
                                      [tf.shape(self.input_seq)[0], args.maxlen, args.candidate_count])
        self.test_logits = self.test_logits[:, -1, :]  # 取最后一个item的得分

        # repeat test
        self.repeat_test_logits = tf.matmul(repeat_seq_emb, tf.transpose(test_item_emb))
        self.repeat_test_logits = tf.reshape(self.repeat_test_logits,
                                             [tf.shape(self.input_seq)[0], args.maxlen, args.candidate_count])
        self.repeat_test_logits = self.repeat_test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)  # 序列表示和正例embedding相乘得分，二维*二维，每个session中每个item对正例的评分
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        self.repeat_pos_logits = tf.reduce_sum(pos_emb * repeat_seq_emb, -1)
        self.repeat_neg_logits = tf.reduce_sum(neg_emb * repeat_seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)),
                              [tf.shape(self.input_seq)[0] * args.maxlen])  # 一维表，每个正例是否为0
        isrepeat = tf.reshape(tf.to_float(self.pos_mask), [tf.shape(self.input_seq)[0] * args.maxlen])
        notrepeat = tf.reshape(tf.to_float(tf.not_equal(self.pos_mask, 1)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget * notrepeat -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget * notrepeat
        ) / tf.reduce_sum(istarget * notrepeat)

        self.repeat_loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.repeat_pos_logits) + 1e-24) * istarget * isrepeat -
            tf.log(1 - tf.sigmoid(self.repeat_neg_logits) + 1e-24) * istarget * isrepeat
        ) / tf.reduce_sum(istarget * isrepeat)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses) + self.repeat_loss

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(
                self.pos_logits - self.neg_logits + self.repeat_pos_logits - self.repeat_neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, time_interval):
        return sess.run([self.test_logits, self.repeat_test_logits],
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False,
                         self.time_interval: time_interval})

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
