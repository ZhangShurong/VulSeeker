# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import csv
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import numba
import itertools
# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
D = 8  # dimensional
P = 64  # embedding_size
B = 10  # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
epochs = 100
is_debug = True

data_folder="./features/"
mydata_folder="./data/"
# INDEX = mydata_folder + "index_mydata.csv"
INDEX = data_folder + "index.csv"
PREFIX = ""

test_file = os.path.join(mydata_folder, "test"+PREFIX+".csv")

TEST_TFRECORD="data/TFrecord/test_gemini_data_"+"5430"+PREFIX+".tfrecord"

print (TEST_TFRECORD)

# ==================== load the function pairs list ===================
#       1.   load_dataset()      load the pairs list for learning, which are
#                                in train.csv, valid.csv, test.csv .
#       1-1. load_csv_as_pair()  process each csv file.
# =====================================================================
def load_dataset():
    """ load the pairs list for training, testing, validing
    """
    test_pair, test_label = load_csv_as_pair(test_file)

    return  test_pair, test_label

def load_csv_as_pair(pair_label_file):
    """ load each csv file, which record the pairs list for learning and its label ( 1 or -1 )
        csv file : uid, uid, 1/-1 eg: 1.1.128, 1.4.789, -1
        pair_dict = {(uid, uid) : -1/1}
    """
    pair_list = []
    label_list = []
    with open(pair_label_file, "r") as fp:
        pair_label = csv.reader(fp)
        for line in pair_label:
            pair_list.append([line[0], line[1]])
            label_list.append(int(line[2]))

    return pair_list, label_list


# ====================== load block info and cfg ======================
#       1.   load_all_data()     load the pairs list for learning, which are
#                                in train.csv, valid.csv, test.csv .
#       1-1. load_block_info()
#       1-2. load_graph()
# =====================================================================
def load_all_data():
    """ load all the real data, including blocks' featrue & functions' cfg using networkx
        uid_graph = {uid: nx_graph}
        feature_dict = {identifier : [[feature_vector]...]}, following the block orders
    """
    uid_graph = {} #保存cfg图,id为版本号+基快地址
    feature_dict = {} #各个版本的所有block特征，id的版本号，值为block集合{基地址：特征}
    # read the direcory list and its ID
    # traversal each record to load every folder's data
    with open(INDEX, "r") as fp:
        for line in csv.reader(fp):
            # index.csv : folder name, identifier
            # eg： openssl-1.0.1a_gcc_4.6_dir, 1.1

            # load_block_info: save all the blocks' feature vector into feature_dict;
            #                  return current file's block number saved into block_num;
            #                  return each function's block id list saved into cur_func_block_dict.
            # 函数地址、  函数地址->函数调用地址
            block_num, cur_func_block_dict = load_block_info(os.path.join(data_folder, line[0], "block_info.csv"),
                                                             feature_dict, line[1])

            if is_debug:
                print ("load cfg ...")
            # load every function's cfg
            load_graph(os.path.join(data_folder, line[0], "adj_info.txt"), block_num, cur_func_block_dict, line[1],
                       uid_graph)

    return uid_graph, feature_dict


def load_my_data():
    """ load all the real data, including blocks' featrue & functions' cfg using networkx
        uid_graph = {uid: nx_graph}
        feature_dict = {identifier : [[feature_vector]...]}, following the block orders
    """
    uid_graph = {} #保存cfg图,id为版本号+基快地址
    feature_dict = {} #各个版本的所有block特征，id的版本号，值为block集合{基地址：特征}
    # read the direcory list and its ID
    # traversal each record to load every folder's data

    # 函数地址、  函数地址->函数调用地址
    block_num, cur_func_block_dict = load_block_info(os.path.join(mydata_folder, "block_info.csv"), feature_dict , 'mydata')

    if is_debug:
        print ("load cfg ...")
    # load every function's cfg
    load_graph(os.path.join(mydata_folder, "adj_info.txt"), block_num, cur_func_block_dict,'mydata',
           uid_graph)

    return uid_graph, feature_dict


def load_block_info(feature_file, feature_dict, uid_prefix):
    """ load all the blocks' feature vector into feature dictionary.
        the two returned values are for next step, loading graph.
        return the block numbers —— using to add the single node of the graph
        return cur_func_blocks_dict —— using to generate every function's cfg（subgraph)
    """
    feature_dict[str(uid_prefix)] = []#版本号：特征矩阵
    cur_func_blocks_dict = {}#函数地址->函数调用地址

    block_uuid = []#函数地址
    line_num = 0
    block_feature_dic = {}#函数地址:[特征矩阵（指令数量）]
    with open(feature_file, "r") as fp:
        if is_debug:
            print (feature_file)
        for line in csv.reader(fp):
            line_num += 1
            # skip the topic line
            #if line_num == 1:
            #    continue
            if line[0] == "":
                continue
            block_uuid.append(str(line[0]))
            # read every bolck's features
            block_feature = [float(x) for x in (line[4:13])]
            del block_feature[6]
            block_feature_dic.setdefault(str(line[0]),block_feature)

            # record each function's block id.
            # for next step to generate the control flow graph
            # so the root block need be add.
            if str(line[2].strip()) not in cur_func_blocks_dict:
                cur_func_blocks_dict[str(line[2].strip())] = [str(line[0])]
            else:
                cur_func_blocks_dict[str(line[2].strip())].append(str(line[0]))
        feature_dict[str(uid_prefix)].append(block_feature_dic)

    return block_uuid, cur_func_blocks_dict



#@numba.jit
def load_graph(graph_file, block_number, cur_func_blocks_dict, uid_prefix, uid_graph):
    """ load all the graph as networkx
    """
    graph = nx.read_edgelist(graph_file, create_using=nx.DiGraph(), nodetype=str)

    # add the missing vertexs which are not in edge_list
    for i in block_number:
        if i not in graph.nodes():
            graph.add_node(i)

    for func_id in cur_func_blocks_dict:
        graph_sub = graph.subgraph(cur_func_blocks_dict[func_id])
        uid = uid_prefix + "." + str(func_id)
        uid_graph[uid] = graph_sub
        # -----------------------可视化cfg图----------------------
        # print('输出网络中的节点...')
        # print(graph_sub.nodes())
        # print('输出网络中的边...')
        # print(graph_sub.edges())
        # print('输出网络中边的数目...')
        # print(graph_sub.number_of_edges())
        # print('输出网络中节点的数目...')
        # print(graph_sub.number_of_nodes())
        # print('给网路设置布局...')
        # pos = nx.shell_layout(graph_sub)
        # nx.draw(graph_sub, pos, with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
        # plt.show()


#@numba.jit
def construct_learning_dataset(uid_pair_list):
    """ Construct pairs dataset to train the model.
        attributes:
            adj_matrix_all  store each pairs functions' graph info, （i,j)=1 present i--》j, others （i,j)=0
            features_all    store each pairs functions' feature map
    """
    print ("     start generate adj matrix pairs...")
    adj_matrix_all_1, adj_matrix_all_2 = generate_adj_matrix_pairs(uid_pair_list)

    print ("     start generate features pairs...")
    ### !!! record the max number of a function's block
    features_all_1, features_all_2, max_size, num1, num2 = generate_features_pair(uid_pair_list)

    return adj_matrix_all_1, adj_matrix_all_2, features_all_1, features_all_2, num1, num2, max_size


#@numba.jit
def generate_adj_matrix_pairs(uid_pair_list):
    """ construct all the function pairs' cfg matrix.
    """
    adj_matrix_all_1 = []
    adj_matrix_all_2 = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print ("         %04d martix, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1]))
        adj_matrix_pair = []
        # each pair process two function
        key_0= 'mydata' + "." + str(uid_pair[0])
        for eachKey in uid_graph_1.keys():
            print(eachKey)
        graph = uid_graph_1[key_0]

        # pos = nx.shell_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
        # plt.show()
        # origion_adj_1 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_1.resize(size, size, refcheck=False)
        # adj_matrix_all_1.append(origion_adj_1.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        #扁平化列表
        adj_matrix_all_1.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        select_list = uid_pair[1].split('.')
        if (len(select_list) >= 2):
            graph = uid_graph[uid_pair[1]]   #解决标签为1的情况
        else:
            key_1 = "mydata" + "." + uid_pair[1]   #标签为-1的情况
            graph = uid_graph_1[key_1]
        # origion_adj_2 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_2.resize(size, size, refcheck=False)
        # adj_matrix_all_2.append(origion_adj_2.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        adj_matrix_all_2.append(",".join(list(itertools.chain.from_iterable(adj_str))))

    return adj_matrix_all_1, adj_matrix_all_2


#@numba.jit
def convert_graph_to_adj_matrix(graph):
    """ convert the control flow graph as networkx to a adj matrix （v_num * v_num).
        1 present an edge; 0 present no edge
    """
    node_list = graph.nodes()
    adj_matrix = []

    # get all the block id in the cfg
    # construct a v_num * v_num adj martix
    for u in node_list:
        # traversal each block's edgd list,to add the
        u_n = graph.neighbors(u)
        neighbors = []
        for tmp in u_n:
            neighbors.append(tmp)
        node_adj = []
        for v in node_list:
            if v in neighbors:
                node_adj.append(1)
            else:
                node_adj.append(0)
        adj_matrix.append(node_adj)
    # print adj_matrix
    return adj_matrix


#@numba.jit
def generate_features_pair(uid_pair_list):
    """ Construct each function pairs' block feature map.
    """
    node_vector_all_1 = []
    node_vector_all_2 = []
    num1 = []
    num2 = []
    node_length = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print ("         %04d feature, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1]))
        node_vector_pair = []
        # each pair process two function
        uid = 'mydata' + "." + str(uid_pair[0])

        node_list = uid_graph_1[uid].nodes()
        uid_prefix = uid.rsplit('.', 1)[0]  # 从右边第一个'.'分界，分成两个字符串 即 identifier, function_id
        node_vector = []
        for node in node_list:
            node_vector.append(feature_dict_1[str(uid_prefix)][0][node])
        node_length.append(len(node_vector))
        num1.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        node_vector_all_1.append(",".join(list(itertools.chain.from_iterable(node_str))))

        select_list = uid_pair[1].split('.')
        node_vector = []
        if (len(select_list) >= 2):
            uid = uid_pair[1]
            node_list = uid_graph[uid].nodes()
            uid_prefix = uid.rsplit('.', 1)[0]
            for node in node_list:
                node_vector.append(feature_dict[str(uid_prefix)][0][node])
        else:
            uid = "mydata" + "." + uid_pair[1]   #标签为-1的情况
            node_list = uid_graph_1[uid].nodes()
            uid_prefix = uid.rsplit('.', 1)[0]  # 从右边第一个'.'分界，分成两个字符串 即 identifier, function_id
            for node in node_list:
                node_vector.append(feature_dict_1[str(uid_prefix)][0][node])
        node_length.append(len(node_vector))
        num2.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        node_vector_all_2.append(",".join(list(itertools.chain.from_iterable(node_str))))

    num1_re = np.array(num1)
    num2_re = np.array(num2)
    #num1_re = num1_arr.astype(np.string_)
    #num2_re = num2_arr.astype(np.string_)
    #(100000)(100000)()(100000,)(100000,)
    return node_vector_all_1, node_vector_all_2, np.max(node_length),num1_re,num2_re

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



# ========================== the main function ========================
#       1.  load_dataset()  load the train, valid, test csv file.
#       2.  load_all_data() load the origion data, including block info, cfg by networkx.
#       3.  construct_learning_dataset() combine the csv file and real data, construct training dataset.
# =====================================================================
# 1. load the train, valid, test csv file.
data_time = time.time()
test_pair, test_label = load_dataset()
print ("1. loading pairs list time", time.time() - data_time, "(s)")

# 2. load the origion data, including block info, cfg by networkx.
graph_time = time.time()
uid_graph_1, feature_dict_1 = load_my_data()
uid_graph, feature_dict = load_all_data()
print ("2. loading graph data time", time.time() - graph_time, "(s)")

# 3. construct training dataset.
cons_time = time.time()


# ========================== clean memory ========================
# del train_pair, train_adj_matrix_1,train_adj_matrix_2,train_feature_map_1,train_feature_map_2,train_max

# ========================== clean memory ========================
# del valid_pair, valid_adj_matrix_1,valid_adj_matrix_2,valid_feature_map_1,valid_feature_map_2,valid_max

# ======================= construct test data =====================
test_adj_matrix_1, test_adj_matrix_2, test_feature_map_1, test_feature_map_2,test_num1, test_num2, test_max \
    = construct_learning_dataset(test_pair)
# ========================== store in pickle ========================
node_list = np.linspace(test_max,test_max, len(test_label),dtype=int)
writer = tf.python_io.TFRecordWriter(TEST_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8 in itertools.izip(
        test_label, test_adj_matrix_1, test_adj_matrix_2, test_feature_map_1, test_feature_map_2,
        test_num1, test_num2, node_list):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                'adj_matrix_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                'adj_matrix_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                'feature_map_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                'feature_map_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
                'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item7])),
                'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
