# coding=utf-8
import logging
import tensorflow as tf

# Parametersembedding_size
num_steps = 500
display_step = 100
batch_size = 10

# Network Parameters
dimension = 7
embedding_size = 54
learning_rate = 0.0001

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
        list_train_num1, list_train_num2, list_train_max = read_and_decode(TRAIN_TFRECORD)
    
    batch_train_label, batch_train_adj_matrix_1, batch_train_adj_matrix_2, batch_train_feature_map_1, \
        batch_train_feature_map_2, batch_train_num1, batch_train_num2, batch_train_max  \
        = tf.train.batch([list_train_label, list_train_adj_matrix_1, list_train_adj_matrix_2, list_train_feature_map_1,
                      list_train_feature_map_2, list_train_num1, list_train_num2, list_train_max],
                      batch_size=batch_size, capacity=100)
    


def train_model(dataset_path, model_path, log_path):
    logging.info("Reading dataset.")
    load_data()
    train()


def main():
    pass

if __name__ == '__main__':
    main()