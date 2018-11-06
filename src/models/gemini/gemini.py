# coding=utf-8
import logging
import tensorflow as tf
import os
import time
import numpy as np

# Parametersembedding_size
num_steps = 500
display_step = 100
batch_size = 10

# Network Parameters
dimension = 7
embedding_size = 54
learning_rate = 0.0001

model_conf = {}
model_conf['train_tfrecord'] = ''
model_conf['test_tfrecord'] = ''
model_conf['valid_tfrecord'] = ''
model_conf['log_dir'] = ''
model_conf['model_dir'] = ''

def load_data():
    pass

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'cfg_1': tf.FixedLenFeature([], tf.string),
        'cfg_2': tf.FixedLenFeature([], tf.string),
        'fea_1': tf.FixedLenFeature([], tf.string),
        'fea_2': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.int64),
        'num2': tf.FixedLenFeature([], tf.int64),
        'max': tf.FixedLenFeature([], tf.int64)})

    label = tf.cast(features['label'], tf.int32)
    graph_1 = features['cfg_1']
    graph_2 = features['cfg_2']
    num1 = tf.cast(features['num1'], tf.int32)
    feature_1 = features['fea_1']
    num2 =  tf.cast(features['num2'], tf.int32)
    feature_2 = features['fea_2']
    max_num = tf.cast(features['max'], tf.int32)

    return label, graph_1, graph_2, feature_1, feature_2, num1, num2, max_num

def contrastive_loss(labels, distance):
    #    tmp= y * tf.square(d)
    #    #tmp= tf.mul(y,tf.square(d))
    #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
    #    return tf.reduce_sum(tmp +tmp2)/B/2
    #    print "contrastive_loss", y,
    loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
    return loss

def cal_distance(model1, model2):
    a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1,(1,-1)),
                                                             tf.reshape(model2,(1,-1))],0),0),(batch_size, embedding_size)),1,keep_dims=True)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1), 1, keep_dims=True))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2), 1, keep_dims=True))
    distance = a_b/tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm,(1,-1)),
                                                        tf.reshape(b_norm,(1,-1))],0),0),(batch_size,1))
    return distance
    
def structure2vec(mu_prev, adj_matrix, x, name="structure2vec"):
    with tf.variable_scope(name):
        # n层全连接层 + n-1层激活层
        # n层全连接层  将v_num个P*1的特征汇总成P*P的feature map
        # 初始化P1,P2参数矩阵，截取的正态分布模式初始化  stddev是用于初始化的标准差
        # 合理的初始化会给网络一个比较好的训练起点，帮助逃脱局部极小值（or 鞍点）
        W_1 = tf.get_variable('W_1', [dimension, embedding_size], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_1 = tf.get_variable('P_1', [embedding_size, embedding_size], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_2 = tf.get_variable('P_2', [embedding_size, embedding_size], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        L = tf.reshape(tf.matmul(adj_matrix, mu_prev, transpose_a=True), (-1, embedding_size))  # v_num * P
        S = tf.reshape(tf.matmul(tf.nn.relu(tf.matmul(L, P_2)), P_1), (-1, embedding_size))

        return tf.tanh(tf.add(tf.reshape(tf.matmul(tf.reshape(x, (-1, dimension)), W_1), (-1, embedding_size)), S))


def structure2vec_net(adj_matrix, x, v_num):
    with tf.variable_scope("structure2vec_net") as structure2vec_net:
        B_mu_5 = tf.Variable(tf.zeros(shape = [0, embedding_size]), trainable=False)
        w_2 = tf.get_variable('w_2', [embedding_size, embedding_size], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        for i in range(batch_size):
            cur_size = tf.to_int32(v_num[i][0])
            mu_0 = tf.reshape(tf.zeros(shape = [cur_size, embedding_size]),(cur_size, embedding_size))
            adj = tf.slice(adj_matrix[i], [0, 0], [cur_size, cur_size])
            fea = tf.slice(x[i], [0,0], [cur_size, dimension])
            mu_1 = structure2vec(mu_0, adj, fea)
            structure2vec_net.reuse_variables()
            mu_2 = structure2vec(mu_1, adj, fea)
            mu_3 = structure2vec(mu_2, adj, fea)
            mu_4 = structure2vec(mu_3, adj, fea)
            mu_5 = structure2vec(mu_4, adj, fea)
            B_mu_5 = tf.concat([B_mu_5, tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, embedding_size)), w_2)],0)
        return B_mu_5

def get_batch(label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num):
    y = np.reshape(label, [batch_size, 1])
    v_num_1 = []
    v_num_2 = []
    for i in range(batch_size):
        v_num_1.append([int(num1[i])])
        v_num_2.append([int(num2[i])])
    graph_1 = []
    graph_2 = []
    for i in range(batch_size):
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
    
    feature_1 = []
    feature_2 = []
    for i in range(batch_size):
        feature_arr = np.array(feature_str1[i].split(','))
        feature_ori = feature_arr.astype(np.float32)
        feature_vec1 = np.resize(feature_ori,(np.max(v_num_1), dimension))
        feature_1.append(feature_vec1)

        feature_arr = np.array(feature_str2[i].split(','))
        feature_ori= feature_arr.astype(np.float32)
        feature_vec2 = np.resize(feature_ori,(np.max(v_num_2), dimension))
        feature_2.append(feature_vec2)

    return y, graph_1, graph_2, feature_1, feature_2, v_num_1, v_num_2
    
def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.5
    for i in xrange(len(prediction)):
        if labels[i][0] == 1:
            if prediction[i][0] > threshold:
                accu += 1.0
        else:
            if prediction[i][0] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc
def calculate_auc(labels, predicts):
    fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
    AUC = auc(fpr, tpr)
    print "auc : ",AUC
    return AUC

def train():
    labels = tf.placeholder(tf.float32, shape=([batch_size, 1]), name= 'label')
    dropout_f = tf.placeholder(tf.float32)
    v_num_left = tf.placeholder(tf.float32, shape=[batch_size, 1], name='v_num_left')
    graph_left = tf.placeholder(tf.float32, shape=([batch_size, None, None]), name='graph_left')
    feature_left = tf.placeholder(tf.float32, shape=([batch_size, None, dimension]), name='feature_left')
    
    v_num_right = tf.placeholder(tf.float32, shape=[batch_size, 1], name='v_num_right')
    graph_right = tf.placeholder(tf.float32, shape=([batch_size, None, None]), name='graph_right')
    feature_right = tf.placeholder(tf.float32, shape=([batch_size, None, dimension]), name='feature_right')

    with tf.variable_scope('siamese') as siamese:
        model1 = structure2vec_net(graph_left, feature_left, v_num_left)
        siamese.reuse_variables()
        model2 = structure2vec_net(graph_right, feature_right, v_num_right)
    
    dis = cal_distance(model1, model2)
    loss = contrastive_loss(labels, dis)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    list_train_label, list_train_adj_matrix_1, list_train_adj_matrix_2, list_train_feature_map_1, list_train_feature_map_2, \
        list_train_num1, list_train_num2, list_train_max = read_and_decode(model_conf['train_tfrecord'])
    
    batch_train_label, batch_train_adj_matrix_1, batch_train_adj_matrix_2, batch_train_feature_map_1, \
        batch_train_feature_map_2, batch_train_num1, batch_train_num2, batch_train_max  \
        = tf.train.batch([list_train_label, list_train_adj_matrix_1, list_train_adj_matrix_2, list_train_feature_map_1,
                      list_train_feature_map_2, list_train_num1, list_train_num2, list_train_max],
                      batch_size=batch_size, capacity=100)
    
    list_valid_label, list_valid_adj_matrix_1, list_valid_adj_matrix_2, list_valid_feature_map_1, list_valid_feature_map_2, \
        list_valid_num1, list_valid_num2, list_valid_max = read_and_decode(model_conf['valid_tfrecord'])

    batch_valid_label, batch_valid_adj_matrix_1, batch_valid_adj_matrix_2, batch_valid_feature_map_1, \
        batch_valid_feature_map_2, batch_valid_num1, batch_valid_num2, batch_valid_max  \
        = tf.train.batch([list_valid_label, list_valid_adj_matrix_1, list_valid_adj_matrix_2, list_valid_feature_map_1,
        list_valid_feature_map_2, list_valid_num1, list_valid_num2, list_valid_max],
        batch_size=batch_size, capacity=100)

    list_test_label, list_test_adj_matrix_1, list_test_adj_matrix_2, list_test_feature_map_1, list_test_feature_map_2, \
        list_test_num1, list_test_num2, list_test_max = read_and_decode(model_conf['test_tfrecord'])
    
    batch_test_label, batch_test_adj_matrix_1, batch_test_adj_matrix_2, batch_test_feature_map_1, \
        batch_test_feature_map_2, batch_test_num1, batch_test_num2, batch_test_max  \
        = tf.train.batch([list_test_label, list_test_adj_matrix_1, list_test_adj_matrix_2, list_test_feature_map_1,
        list_test_feature_map_2, list_test_num1, list_test_num2, list_test_max],
        batch_size=batch_size, capacity=100)
    
    init_opt = tf.global_variables_initializer()
    saver = tf.train.Saver()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=tf_config) as sess:
        writer = tf.summary.FileWriter(model_conf['log_dir'], sess.graph)
        sess.run(init_opt)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    iter=0
    while iter < 100:
        iter += 1
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(50000 / batch_size)
        start_time = time.time()
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
            
            print '     %d    tr_acc %0.2f'%(i, tr_acc)
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        duration = time.time() - start_time

        if iter% 50000 == 0:
            # validing model
            avg_loss = 0.
            avg_acc = 0.
            valid_start_time = time.time()
            valid_num = 5000
            for m in range(int(valid_num / batch_size)):
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
                print '     tr_acc %0.2f'%(tr_acc)

            duration = time.time() - valid_start_time
            print 'valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
                iter, duration, avg_loss / (int(valid_num / batch_size)), avg_acc / (int(valid_num / B)))
            saver.save(sess, model_conf['model_dir'] + os.sep + "gemini-model_"+str(iter)+".ckpt")

            total_labels = []
            total_predicts = []
            avg_loss = 0.
            avg_acc = 0.
            test_num = 5000
            test_total_batch = int(test_num / batch_size)
            start_time = time.time()
            # Loop over all batches
            # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num
            for m in range(test_total_batch):
                test_label, test_adj_matrix_1, test_adj_matrix_2, \
                test_feature_map_1, test_feature_map_2, test_num1, test_num2, test_max = sess.run(
                    [batch_test_label, batch_test_adj_matrix_1, batch_test_adj_matrix_2, batch_test_feature_map_1,
                     batch_test_feature_map_2, batch_test_num1, batch_test_num2, batch_test_max])
                y, graph_1, graph_2, feature_1, feature_2, v_num_1, v_num_2 \
                    = get_batch(test_label, test_adj_matrix_1, test_adj_matrix_2,
                                test_feature_map_1, test_feature_map_2, test_num1, test_num2, test_max)
                predict = dis.eval(
                    feed_dict={graph_left: graph_1, feature_left: feature_1, v_num_left: v_num_1, graph_right: graph_2,
                               feature_right: feature_2, v_num_right: v_num_2, labels: y, dropout_f: 1.0})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                total_labels.append(y)
                total_predicts.append(predict)
                print '     %d    tr_acc %0.2f' % (m, tr_acc)
            duration = time.time() - start_time
            total_labels = np.reshape(total_labels, (-1))
            total_predicts = np.reshape(total_predicts, (-1))
            print "label : ", total_labels
            print "predict: ", total_predicts
            print calculate_auc(total_labels, total_predicts)
            print 'test set, time, %f, loss, %0.5f, acc, %0.2f' % (duration, avg_loss / test_total_batch, avg_acc / test_total_batch)

# 保存模型
    save_path = saver.save(sess, model_conf['model_dir'] + os.sep + "gemini-model_final.ckpt")
    print save_path

    coord.request_stop()
    coord.join(threads)

def train_model(dataset_path, model_path, log_path):
    logging.info("Reading dataset... ...")
    model_conf['train_tfrecord'] = dataset_path + os.sep + 'train.tfrecord'
    model_conf['test_tfrecord'] = dataset_path + os.sep + 'test.tfrecord'
    model_conf['valid_tfrecord'] = dataset_path + os.sep + 'valid.tfrecord'
    model_conf['model_dir'] = model_path
    model_conf['log_dir'] = log_path
    load_data()
    train()


def main():
    pass

if __name__ == '__main__':
    main()