"""
Implementation of End to End Memory Networks (MenMM)

Reference
1. https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf
2. https://domluna.me/memory-networks/

cuteboydot@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import pickle
import numpy as np
import tensorflow as tf
from utils import data_loader

np.core.arrayprint._line_width = 1000
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# file_data_train = "./data/qa1_single-supporting-fact_train.txt"
# file_data_test = "./data/qa1_single-supporting-fact_test.txt"
file_data_train = "./data/qa16_basic-induction_train.txt"
file_data_test = "./data/qa16_basic-induction_test.txt"

file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_list_train = "./data/data_list_train.bin"
file_data_idx_list_train = "./data/data_idx_list_train.bin"
file_data_list_test = "./data/data_list_test.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
dir_summary = "./model/summary/"

pre_trained = 0
my_device = "/cpu:0"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(now)

if pre_trained == 0:
    print("Load data file & make vocabulary...")

    data_list_train, data_list_test, data_idx_list_train, data_idx_list_test, rdic, dic, max_len = \
        data_loader(file_data_train, file_data_test)
    max_story_len = max_len[0]
    max_words_len = max_len[1]

    # save dictionary
    with open(file_data_list_train, 'wb') as handle:
        pickle.dump(data_list_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list_train, 'wb') as handle:
        pickle.dump(data_idx_list_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_list_test, 'wb') as handle:
        pickle.dump(data_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list_test, 'wb') as handle:
        pickle.dump(data_idx_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic, 'wb') as handle:
        pickle.dump(rdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_max_len, 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Load vocabulary from model file...")

    # load dictionary
    with open(file_data_list_train, 'rb') as handle:
        data_list_train = pickle.load(handle)
    with open(file_data_idx_list_train, 'rb') as handle:
        data_idx_list_train = pickle.load(handle)
    with open(file_data_list_test, 'rb') as handle:
        data_list_test = pickle.load(handle)
    with open(file_data_idx_list_test, 'rb') as handle:
        data_idx_list_test = pickle.load(handle)
    with open(file_dic, 'rb') as handle:
        dic = pickle.load(handle)
    with open(file_rdic, 'rb') as handle:
        rdic = pickle.load(handle)
    with open(file_max_len, 'rb') as handle:
        max_len = pickle.load(handle)
    max_story_len = max_len[0]
    max_words_len = max_len[1]

print("len(data_list_train) = %d" % len(data_idx_list_train))
print("len(data_list_test) = %d" % len(data_idx_list_test))
print("len(dic) = %d" % len(dic))
print("max_story_len = %d" % max_story_len)
print("max_words_len = %d" % max_words_len)
print()

print("data_list_train[0] example")
print(data_list_train[0])
print("data_idx_list_train[0] example")
print(data_idx_list_train[0])
print("rdic example")
print(rdic[:20])
print()


def generate_batch(size):
    assert size <= len(data_idx_list_train)

    data_story = np.zeros((size, max_story_len, max_words_len), dtype=np.int)
    data_story_idx = np.zeros((size, max_story_len), dtype=np.int)
    data_question = np.zeros((size, max_words_len), dtype=np.int)
    data_answer = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_train)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        sto = data_idx_list_train[idx][0]
        que = data_idx_list_train[idx][1]
        ans = data_idx_list_train[idx][2]

        for b in range(len(sto)):
            m = sto[b] + [dic["<nil>"]] * (max_words_len - len(sto[b]))
            data_story[a][b] = m
            data_story_idx[a][b] = b+1
        data_question[a] = que + [dic["<nil>"]] * (max_words_len - len(que))
        data_answer[a] = ans

    return data_story, data_story_idx, data_question, data_answer


def generate_test_batch(size):
    assert size <= len(data_idx_list_test)

    data_story = np.zeros((size, max_story_len, max_words_len), dtype=np.int)
    data_story_idx = np.zeros((size, max_story_len), dtype=np.int)
    data_question = np.zeros((size, max_words_len), dtype=np.int)
    data_answer = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_test)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        sto = data_idx_list_test[idx][0]
        que = data_idx_list_test[idx][1]
        ans = data_idx_list_test[idx][2]

        for b in range(len(sto)):
            m = sto[b] + [dic["<nil>"]] * (max_words_len - len(sto[b]))
            data_story[a][b] = m
            data_story_idx[a][b] = b+1
        data_question[a] = que + [dic["<nil>"]] * (max_words_len - len(que))
        data_answer[a] = ans

    return data_story, data_story_idx, data_question, data_answer


MAX_STORY_LEN = max_story_len
MAX_WORDS_LEN = max_words_len
SIZE_VOC = len(rdic)
SIZE_EMBED_DIM = 20
SIZE_HOP = 3


def position_encoding():
    pos_enc = np.ones((SIZE_EMBED_DIM, MAX_WORDS_LEN), dtype=np.float32)

    for a in range(1, SIZE_EMBED_DIM):
        for b in range(1, MAX_WORDS_LEN):
            pos_enc[a][b] = (1 - b / MAX_WORDS_LEN) - (a / SIZE_EMBED_DIM) * (1 - 2 * b / MAX_WORDS_LEN)
    pos_enc = np.transpose(pos_enc)
    return pos_enc


with tf.Graph().as_default():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Build Graph...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        with tf.device(my_device):

            story = tf.placeholder(tf.int32, [None, MAX_STORY_LEN, MAX_WORDS_LEN], name="story")
            stroy_idx = tf.placeholder(tf.int32, [None, MAX_STORY_LEN], name="stroy_idx")
            question = tf.placeholder(tf.int32, [None, MAX_WORDS_LEN], name="question")
            answer = tf.placeholder(tf.int32, [None, ], name="answer")
            one_hot_answer = tf.one_hot(answer, SIZE_VOC, name="one_hot_answer")

            global_step = tf.Variable(0, name="global_step", trainable=False)
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

            # Position Encoding
            pe = position_encoding()

            embed_nil = tf.zeros([1, SIZE_EMBED_DIM], name="nil_emb")

            A_tmp = tf.Variable(tf.random_normal([SIZE_VOC - 1, SIZE_EMBED_DIM], mean=0, stddev=0.1))
            A_tmp2 = tf.concat([embed_nil, A_tmp], axis=0)
            A = tf.Variable(A_tmp2, name="A")

            # B_tmp = tf.Variable(tf.random_normal([SIZE_VOC - 1, SIZE_EMBED_DIM], mean=0, stddev=0.1))
            # B_tmp2 = tf.concat([embed_nil, A_tmp], axis=0)
            # B = tf.Variable(A_tmp2, name="B")

            C_list = []
            C_tmp = tf.Variable(tf.random_normal([SIZE_VOC - 1, SIZE_EMBED_DIM], mean=0, stddev=0.1))
            C_tmp2 = tf.concat([embed_nil, C_tmp], axis=0)
            for k in range(SIZE_HOP):
                with tf.variable_scope("hop_%d" % k):
                    C_list.append(tf.Variable(C_tmp2, name="C"))

            # Temporal Encoding
            # [MAX_STORY_LEN+1, SIZE_EMBED_DIM]
            # [0, :] -> dummy(zero-value) sentence idx
            # [1:, :] -> story sentence idx
            T_A_tmp = tf.Variable(tf.random_normal([MAX_STORY_LEN, SIZE_EMBED_DIM], mean=0, stddev=0.1))
            T_A_tmp_2 = tf.concat([embed_nil, T_A_tmp], axis=0)
            T_A = tf.Variable(T_A_tmp_2, name="T_A")

            T_C_tmp = tf.Variable(tf.random_normal([MAX_STORY_LEN, SIZE_EMBED_DIM], mean=0, stddev=0.1))
            T_C_tmp_2 = tf.concat([embed_nil, T_C_tmp], axis=0)
            T_C = tf.Variable(T_A_tmp_2, name="T_C")


            def forward(batch_story, batch_question):
                u_list = []
                p_list = []
                o_list = []

                # normal case
                # embed_B = tf.nn.embedding_lookup(B, batch_question, name="embed_B")   # [batch, words, embed]

                # adjacent weight sharing scheme case
                embed_B = tf.nn.embedding_lookup(A, batch_question, name="embed_B")     # [batch, words, embed]
                embed_B = embed_B * pe

                u0 = tf.reduce_sum(embed_B, axis=1)  # [batch, embed]
                u0 = tf.reshape(u0, [-1, SIZE_EMBED_DIM, 1])  # [batch, embed, 1]
                u_list.append(u0)

                for k in range(SIZE_HOP):
                    if k == 0:
                        embed_A = tf.nn.embedding_lookup(A, batch_story)                # [batch, story, words, embed]
                        embed_A = embed_A * pe
                    else:
                        # adjacent weight sharing scheme case
                        embed_A = tf.nn.embedding_lookup(C_list[k - 1], batch_story)    # [batch, story, words, embed]

                    embed_A = tf.reduce_sum(embed_A, axis=2)                            # [batch, story, embed]
                    embed_T_A = tf.nn.embedding_lookup(T_A, stroy_idx)                  # [batch, story, embed]

                    m = tf.add(embed_A, embed_T_A)                                      # [batch, story, embed]
                    u = u_list[-1]                                                      # [batch, embed, 1]
                    p = tf.matmul(m, u)                                                 # [batch, story, 1]
                    attention_prob = tf.nn.softmax(tf.reduce_sum(p, axis=2))            # [batch, story]

                    embed_C = tf.nn.embedding_lookup(C_list[k], batch_story)            # [batch, story, words, embed]
                    embed_C = tf.reduce_sum((embed_C * pe), axis=2)                     # [batch, story, embed]
                    embed_T_C = tf.nn.embedding_lookup(T_C, stroy_idx)                  # [batch, story, embed]

                    c = tf.add(embed_C, embed_T_C)                                      # [batch, story, embed]
                    c = tf.transpose(c, [0, 2, 1])                                      # [batch, embed, story]

                    o = tf.matmul(c, p)                                                 # [batch, embed, 1]

                    u_next = u + o

                    p_list.append(attention_prob)
                    u_list.append(u_next)
                    o_list.append(o)

                u_out = tf.reshape(u_list[-1], [-1, SIZE_EMBED_DIM])                    # [batch, embed]
                w_out = tf.transpose(C_list[-1], [1, 0])                                # [embed, voc]

                # linear start
                a_ = tf.matmul(u_out, w_out)                                            # [batch, voc]

                # non-linear start
                # a_ = tf.nn.softmax(a_)

                attention_map = tf.convert_to_tensor(p_list)                            # [hop, batch, story]
                attention_map = tf.transpose(attention_map, [1, 2, 0])                  # [batch, story, hop]
                return a_, attention_map


            hypothesis, attention = forward(story, question)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                           labels=tf.cast(one_hot_answer, tf.float32))
            loss = tf.reduce_sum(loss, name="loss")

            predict = tf.cast(tf.argmax(hypothesis, 1), tf.int32)
            correct = tf.equal(answer, predict)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            variables = [A, T_A, T_C] + C_list
            grads_and_vars = opt.compute_gradients(loss, variables)

            # clip by norm
            grads_and_vars = [(tf.clip_by_norm(g, 40.0), v) for g, v in grads_and_vars]

            # add noise
            grads_and_vars = [(tf.add(g, tf.random_normal(tf.shape(g), stddev=0.001)), v) for g, v in grads_and_vars]

            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            out_dir = os.path.abspath(os.path.join("./model", timestamp))
            print("LOGDIR = %s" % out_dir)
            print()

        # Summaries
        loss_summary = tf.summary.scalar("loss", loss)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_dir = os.path.join(out_dir, "summary", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model-step")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        if pre_trained == 2:
            print("Restore model file...")
            file_model = "./model/2017-11-09 16:55/checkpoints/"
            saver.restore(sess, tf.train.latest_checkpoint(file_model))

        BATCHS = 50
        BATCHS_TEST = len(data_idx_list_test)
        EPOCHS = 300
        STEPS = int(len(data_idx_list_train) / BATCHS)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        print(now)
        print("Train start!!")

        loop_step = 0
        for epoch in range(EPOCHS):
            for step in range(STEPS):
                data_s, data_s_idx, data_q, data_a = generate_batch(BATCHS)

                feed_dict = {
                    story: data_s,
                    stroy_idx: data_s_idx,
                    question: data_q,
                    answer: data_a
                }

                _, batch_loss, batch_acc, train_sum, g_step = \
                    sess.run([train_op, loss, accuracy, train_summary_op, global_step], feed_dict)

                if loop_step % 50 == 0:
                    train_summary_writer.add_summary(train_sum, g_step)

                if loop_step % 200 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] step[%05d] glob_step[%05d] - batch_loss:%.4f, batch_acc:%.4f (%s) " %
                          (epoch, loop_step, g_step, batch_loss, batch_acc, now))

                # test
                if loop_step % 200 == 0:
                    data_s, data_s_idx, data_q, data_a = generate_test_batch(BATCHS_TEST)

                    feed_dict = {
                        story: data_s,
                        stroy_idx: data_s_idx,
                        question: data_q,
                        answer: data_a
                    }

                    pred, test_loss, test_acc, test_sum, g_step = \
                        sess.run([predict, loss, accuracy, test_summary_op, global_step], feed_dict)
                    test_summary_writer.add_summary(test_sum, g_step)

                if loop_step % 1000 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] step[%05d] glob_step[%05d] - test_loss:%.4f, test_acc:%.4f (%s)" %
                          (epoch, loop_step, g_step, test_loss, test_acc, now))

                if loop_step % 5000 == 0:
                    print("INTERMEDIATES TEST")
                    cnt = 0
                    for a in range(100):
                        print("[%03d]: Y=%02d, Y^=%02d  " % (a, data_a[a], pred[a]), end="")
                        if data_a[a] == pred[a]:
                            cnt += 1
                        if (a + 1) % 5 == 0:
                            print()
                    print("INTERMEDIATES TEST ACC : %02d " % cnt)

                if loop_step % 20000 == 0:
                    saver.save(sess, checkpoint_prefix, global_step=g_step)

                loop_step += 1

        print()
        print("TEST EXAMPLE")
        data_s, data_s_idx, data_q, data_a = generate_test_batch(10)

        feed_dict = {
            story: data_s,
            stroy_idx: data_s_idx,
            question: data_q,
            answer: data_a
        }

        pred, prob = sess.run([predict, attention], feed_dict)

        for a in range(data_s.shape[0]):
            print("Test [%03d]" % (a + 1))
            print("Story")
            for b in range(max_story_len):
                sen = [rdic[w] for w in data_s[a][b]]
                hop_prob = prob[a][b]

                str_sen = "{0:65}".format(str(sen))
                str_att = "{0:60}".format(str(hop_prob))
                print("[%02d] %s, %s" % (b + 1, str_sen, str_att))

            que = [rdic[q] for q in data_q[a]]
            print("Question : %s" % str(que))

            ans = rdic[data_a[a]]
            pre = rdic[pred[a]]
            print("Answer : %s, \t Predict : %s" % (ans, pre))
