# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import csv
import os
import time
import networkx as nx
import numba
import pickle
# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
D = 8  # dimensional
P = 64  # embedding_size
B = 10 # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
max_iter = 100
decay_steps = 10 # 衰减步长
decay_rate = 0.1 # 衰减率
snapshot = 2
is_debug = False

train_num = 100000
valid_num = int(train_num/10)
test_num = int(train_num/10)
PREFIX = "_[0,5]"
TRAIN_TFRECORD="TFrecord/train_gemini_data_"+"100000"+PREFIX+".tfrecord"
TEST_TFRECORD="TFrecord/test_gemini_data_"+"100000"+PREFIX+".tfrecord"
VALID_TFRECORD="TFrecord/valid_gemini_data_"+"100000"+PREFIX+".tfrecord"

# =============== convert the real data to training data ==============
#       1.  construct_learning_dataset() combine the dataset list & real data
#       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
#       1-1-1. convert_graph_to_adj_matrix()    process each cfg
#       1-2. generate_features_pair() traversal list and construct all functions' feature map
# =====================================================================
""" Parameter P = 64, D = 8, T = 7, N = 2,                  B = 10
     X_v = D * 1   <--->   8 * v_num * 10
     W_1 = P * D   <--->   64* 8    W_1 * X_v = 64*1
    mu_0 = P * 1   <--->   64* 1
     P_1 = P * P   <--->   64*64
     P_2 = P * P   <--->   64*64
    mu_2/3/4/5 = P * P     <--->  64*1
    W_2 = P * P     <--->  64*64
"""

def structure2vec(mu_prev, adj_matrix, x, name="structure2vec"):
    """ Construct pairs dataset to train the model.
        构造对数据集来训练模型
    """

    #-------mu v(t+1) = F(x_v, sum{u in N(v)} \mu _u(t) ), \forall v \in V. ------
    #x_v节点属性，令N(v)表示图$g$中节点$v$的相邻顶点集合

    #(3,64)，(3,3)，(3,8)
    with tf.variable_scope(name):
        # n层全连接层 + n-1层激活层
        # n层全连接层  将v_num个P*1的特征汇总成P*P的feature map
        # 初始化P1,P2参数矩阵，截取的正态分布模式初始化  stddev是用于初始化的标准差
        # 合理的初始化会给网络一个比较好的训练起点，帮助逃脱局部极小值（or 鞍点）
        W_1 = tf.get_variable('W_1', [D, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_1 = tf.get_variable('P_1', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_2 = tf.get_variable('P_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        #（8,64），（64,64），（64,64）
        #（3,3）*（3,64）=（3,64）（v_num * P）---sum{u in N(v)} \mu _u(t)
        L = tf.reshape(tf.matmul(adj_matrix, mu_prev, transpose_a=True), (-1, P))  # v_num * P
        #（3,64）*（64,64）=（3,64）

        #sigma(sum{u\in N(v)} \mu _u)={P_1（times) * ReLU(P_2 * ReLU(P_nl))}_{n\ levels}
        S = tf.reshape(tf.matmul(tf.nn.relu(tf.matmul(L, P_2)), P_1), (-1, P))
        #F(x_v, sum_{u\in N(v)} \mu u)=tanh (W_1*x_v + sigma(sum{u\in N(v)} \mu _u)) $$
        #$W_1$是一个$d \times p$的矩阵，$p$是上面提到的嵌入维度。$\sigma$是一个全连接的$n$层神经网络：
        #$P_n$是$p\times p$的矩阵
        return tf.tanh(tf.add(tf.reshape(tf.matmul(tf.reshape(x, (-1, D)), W_1), (-1, P)), S))

def structure2vec_net(adj_matrix, x, v_num):
    with tf.variable_scope("structure2vec_net") as structure2vec_net:
        B_mu_5 = tf.Variable(tf.zeros(shape = [0, P]), trainable=False)
        w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        for i in range(B):
            cur_size = tf.to_int32(v_num[i][0])#假设为3个节点
            # test = tf.slice(B_mu_0[i], [0, 0], [cur_size, P])
            #初始化Structure2vec网络将为每个节点v初始化一个嵌入mu _v(0)
            mu_0 = tf.reshape(tf.zeros(shape = [cur_size, P]),(cur_size,P))#(3,64)
            adj = tf.slice(adj_matrix[i], [0, 0], [cur_size, cur_size])#（3,3）
            fea = tf.slice(x[i],[0,0], [cur_size,D])#（3,8）
    #-----------------Structure2vec为每个节点v初始化一个嵌入mu _v(0)，迭代中更新嵌入（5次）-------------------
            mu_1 = structure2vec(mu_0, adj, fea)  # , name = 'mu_1')
            structure2vec_net.reuse_variables()
            mu_2 = structure2vec(mu_1, adj, fea)  # , name = 'mu_2')
            mu_3 = structure2vec(mu_2, adj, fea)  # , name = 'mu_3')
            mu_4 = structure2vec(mu_3, adj, fea)  # , name = 'mu_4')
            mu_5 = structure2vec(mu_4, adj, fea)  # , name = 'mu_5')
    #图嵌入网络首先为每个节点v \in V计算出一个$p$维的特征嵌入$\mu _v$，
    # 之后$g$的嵌入$\mu _g$由这些顶点嵌入聚合得到。本文简单地使用求和函数，即$\mu _g = \sum _{v\in V} (\mu _v)$
            # B_mu_5.append(tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2))
            B_mu_5 = tf.concat([B_mu_5,tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2)],0)
            #reshape(reduce_sum)==mu _g
            #W_2是另一个p*p的矩阵，用来变换嵌入
        return B_mu_5

def contrastive_loss(labels, distance):
    #    tmp= y * tf.square(d)
    #    #tmp= tf.mul(y,tf.square(d))
    #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
    #    return tf.reduce_sum(tmp +tmp2)/B/2
    #    print "contrastive_loss", y,
    loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
    return loss


def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.5
    for i in range(len(prediction)):
        if labels[i][0] == 1:
            if prediction[i][0] > threshold:
                accu += 1.0
        else:
            if prediction[i][0] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc

#计算余弦距离-相似度
def cal_distance(model1, model2):
    a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1,(1,-1)),
                                                             tf.reshape(model2,(1,-1))],0),0),(B,P)),1,keep_dims=True)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1),1,keep_dims=True))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2),1,keep_dims=True))
    distance = a_b/tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm,(1,-1)),
                                                        tf.reshape(b_norm,(1,-1))],0),0),(B,1))
    return distance

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'adj_matrix_1': tf.FixedLenFeature([], tf.string),
        'adj_matrix_2': tf.FixedLenFeature([], tf.string),
        'feature_map_1': tf.FixedLenFeature([], tf.string),
        'feature_map_2': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.int64),
        'num2': tf.FixedLenFeature([], tf.int64),
        'max': tf.FixedLenFeature([], tf.int64)})

    label = tf.cast(features['label'], tf.int32)

    graph_1 = features['adj_matrix_1']
    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_1 = adj_arr.astype(np.float32)

    graph_2 = features['adj_matrix_2']
    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_2 = adj_arr.astype(np.float32)

    num1 = tf.cast(features['num1'], tf.int32)
    feature_1 = features['feature_map_1']
    #fea_arr = np.reshape((fea_str.split(',')),(node_num,node_num))
    #feature_1 = fea_arr.astype(np.float32)

    num2 =  tf.cast(features['num2'], tf.int32)
    feature_2 = features['feature_map_2']
    #fea_arr = np.reshape(fea_str.split(','),(node_num,node_num))
    #feature_2 = fea_arr.astype(np.float32)

    max_num = tf.cast(features['max'], tf.int32)

    return label, graph_1, graph_2, feature_1, feature_2, num1, num2, max_num


def get_batch( label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num):

    y = np.reshape(label, [B, 1])

    v_num_1 = []
    v_num_2 = []
    for i in range(B):
        v_num_1.append([int(num1[i])])
        v_num_2.append([int(num2[i])])

    # 补齐 martix 矩阵的长度
    graph_1 = []
    graph_2 = []
    for i in range(B):
        graph_arr = np.array(graph_str1[i].split(','))
        graph_adj = np.reshape(graph_arr, (int(num1[i]), int(num1[i])))
        graph_ori1 = graph_adj.astype(np.float32)
        graph_ori1.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        graph_1.append(graph_ori1.tolist())

        graph_arr = np.array(graph_str2[i].split(','))
        graph_adj = np.reshape(graph_arr, (int(num2[i]), int(num2[i])))
        graph_ori2 = graph_adj.astype(np.float32)
        graph_ori2.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        graph_2.append(graph_ori2.tolist())

    # 补齐 feature 列表的长度
    feature_1 = []
    feature_2 = []
    for i in range(B):
        feature_arr = np.array(feature_str1[i].split(','))
        feature_ori = feature_arr.astype(np.float32)
        feature_vec1 = np.resize(feature_ori,(np.max(v_num_1),D))
        feature_1.append(feature_vec1)

        feature_arr = np.array(feature_str2[i].split(','))
        feature_ori= feature_arr.astype(np.float32)
        feature_vec2 = np.resize(feature_ori,(np.max(v_num_2),D))
        feature_2.append(feature_vec2)

    return y, graph_1, graph_2, feature_1, feature_2, v_num_1, v_num_2

# 4.construct the network
# Initializing the variables
# Siamese network major part

# Initializing the variables

init = tf.global_variables_initializer()
global_step = tf.Variable(0, trainable=False)
#每10轮训练后要乘以0.1
learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)

#图嵌入网络参数
#节点数、、特征维度
v_num_left = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_left')#节点数[[16],[11],[12]...]
graph_left = tf.placeholder(tf.float32, shape=([B, None, None]), name='graph_left')#邻接矩阵
feature_left = tf.placeholder(tf.float32, shape=([B, None, D]), name='feature_left')#特征维度

v_num_right = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_right')
graph_right = tf.placeholder(tf.float32, shape=([B, None, None]), name='graph_right')
feature_right = tf.placeholder(tf.float32, shape=([B, None, D]), name='feature_right')

labels = tf.placeholder(tf.float32, shape=([B, 1]), name='gt')

dropout_f = tf.placeholder("float")

with tf.variable_scope("siamese") as siamese:
    model1 = structure2vec_net(graph_left, feature_left, v_num_left)
    siamese.reuse_variables()# 左右网络共享参数
    model2 = structure2vec_net(graph_right, feature_right, v_num_right)

dis = cal_distance(model1, model2)

loss = contrastive_loss(labels, dis)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

list_train_label, list_train_adj_matrix_1, list_train_adj_matrix_2, list_train_feature_map_1, list_train_feature_map_2, \
list_train_num1, list_train_num2, list_train_max = read_and_decode(TRAIN_TFRECORD)
batch_train_label, batch_train_adj_matrix_1, batch_train_adj_matrix_2, batch_train_feature_map_1, \
batch_train_feature_map_2, batch_train_num1, batch_train_num2, batch_train_max  \
    = tf.train.batch([list_train_label, list_train_adj_matrix_1, list_train_adj_matrix_2, list_train_feature_map_1,
                      list_train_feature_map_2, list_train_num1, list_train_num2, list_train_max],
                     batch_size=B, capacity=100)

list_valid_label, list_valid_adj_matrix_1, list_valid_adj_matrix_2, list_valid_feature_map_1, list_valid_feature_map_2, \
list_valid_num1, list_valid_num2, list_valid_max = read_and_decode(VALID_TFRECORD)
batch_valid_label, batch_valid_adj_matrix_1, batch_valid_adj_matrix_2, batch_valid_feature_map_1, \
batch_valid_feature_map_2, batch_valid_num1, batch_valid_num2, batch_valid_max  \
    = tf.train.batch([list_valid_label, list_valid_adj_matrix_1, list_valid_adj_matrix_2, list_valid_feature_map_1,
                      list_valid_feature_map_2, list_valid_num1, list_valid_num2, list_valid_max],
                     batch_size=B, capacity=100)

list_test_label, list_test_adj_matrix_1, list_test_adj_matrix_2, list_test_feature_map_1, list_test_feature_map_2, \
list_test_num1, list_test_num2, list_test_max = read_and_decode(TEST_TFRECORD)
batch_test_label, batch_test_adj_matrix_1, batch_test_adj_matrix_2, batch_test_feature_map_1, \
batch_test_feature_map_2, batch_test_num1, batch_test_num2, batch_test_max  \
    = tf.train.batch([list_test_label, list_test_adj_matrix_1, list_test_adj_matrix_2, list_test_feature_map_1,
                      list_test_feature_map_2, list_test_num1, list_test_num2, list_test_max],
                     batch_size=B, capacity=100)
''''''
init_opt = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init_opt)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Training cycle
    iter=0
    while iter < max_iter:
        iter += 1
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(train_num / B)
        start_time = time.time()
        # Loop over all batches
        # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_nu
        for i in range(total_batch):
            train_label, train_adj_matrix_1, train_adj_matrix_2, train_feature_map_1, train_feature_map_2, \
            train_num1, train_num2, train_max \
                = sess.run([batch_train_label, batch_train_adj_matrix_1, batch_train_adj_matrix_2,
                            batch_train_feature_map_1, batch_train_feature_map_2, batch_train_num1,
                            batch_train_num2, batch_train_max])
            y, graph_1, graph_2, feature_1, feature_2, v_num_1, v_num_2 \
                = get_batch(train_label, train_adj_matrix_1, train_adj_matrix_2, train_feature_map_1,
                            train_feature_map_2, train_num1, train_num2,  train_max)
            _, loss_value, predict = sess.run([optimizer, loss, dis], feed_dict = {
                graph_left: graph_1, feature_left: feature_1,v_num_left: v_num_1, graph_right: graph_2,
                feature_right: feature_2, v_num_right: v_num_2,labels: y, dropout_f: 0.9})
            tr_acc = compute_accuracy(predict, y)
            if is_debug:
                print ('     %d    tr_acc %0.2f'%(i, tr_acc))
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        duration = time.time() - start_time


        if iter%snapshot == 0:
            # validing model
            avg_loss = 0.
            avg_acc = 0.
            valid_start_time = time.time()
            for m in range(int(valid_num / B)):
                valid_label, valid_adj_matrix_1, valid_adj_matrix_2, valid_feature_map_1, valid_feature_map_2,  \
                valid_num1, valid_num2, valid_max \
                    = sess.run([batch_valid_label, batch_valid_adj_matrix_1, batch_valid_adj_matrix_2,
                                batch_valid_feature_map_1, batch_valid_feature_map_2, batch_valid_num1,
                                batch_valid_num2, batch_valid_max])
                y, graph_1, graph_2, feature_1, feature_2, v_num_1, v_num_2 \
                    = get_batch(valid_label, valid_adj_matrix_1, valid_adj_matrix_2, valid_feature_map_1,
                                valid_feature_map_2,valid_num1, valid_num2,  valid_max)
                predict = dis.eval(feed_dict={
                    graph_left: graph_1, feature_left: feature_1, v_num_left: v_num_1, graph_right: graph_2,
                    feature_right: feature_2, v_num_right: v_num_2, labels: y, dropout_f: 0.9})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                if is_debug:
                    print ('     tr_acc %0.2f'%(tr_acc))
            duration = time.time() - valid_start_time
            print ('valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
                iter, duration, avg_loss / (int(valid_num / B)), avg_acc / (int(valid_num / B))))
            saver.save(sess, "./model/gemini-model_"+str(iter)+".ckpt")


    # 保存模型
    save_path = saver.save(sess, "./model/gemini-model_final.ckpt")
    print (save_path)

    coord.request_stop()
    coord.join(threads)
