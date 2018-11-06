# _*_ coding: utf-8 _*_

import csv
import os
import glob
import random
## id,func_name,func_id,block_id_in_func,numeric constants,string constants,No. of transfer instructions,No. of calls,No. of instructinos,No. of arithmetic instructions,No. of logic instructions,No. of offspring,betweenness centrality
#生成训练集中函数block的数量
# block_num_min< block_num <= block_num_max
# if block_num_max = -1, 忽略这一设置,不考虑block数量
block_num_min = 0
block_num_max = -1
PREFIX = ""


#正例负例数量
pos_num = 1
neg_num = 1
train_dataset_num = 2500
test_dataset_num = int(train_dataset_num/10)
vaild_dataset_num = int(train_dataset_num/10)

dir_folder = "./data/"
ori_dir_folder= "./features/"
func_list_file = dir_folder + "function_list"+PREFIX+".csv"
func_list_ori = ori_dir_folder + "function_list"+PREFIX+".csv"
train_file = dir_folder + "train"+PREFIX+".csv"
test_file = dir_folder + "test"+PREFIX+".csv"
vaild_file = dir_folder + "vaild"+PREFIX+".csv"


# function name, program name, block num, version uid list,
func_list_fp = open(func_list_file, "w")
with open(dir_folder + "block_info.csv", "r") as fp:
    print (dir_folder  + "block_info.csv")
    #func_name_set = set()
    block_num = 0
    func_name = ""
    func_uuid = ""
    for line in csv.reader(fp):
        #00x0L,chmod_or_fchmod,00x0L,0,1,0,0,0,9,0,0,2,0.0

        if line[1] == func_name:
            block_num = block_num + 1
        else :
            print ("new function : ",line[1],line[2])
            if not func_name == "" :
                #print "             ",k,func_name,func_uuid
                #func_name_set.remove(func_name)
                func_list_fp.write(func_name.strip()+","+str(block_num)+","+func_uuid.strip()+"\n")
            #func_name_set.add(line[1])
            block_num = 1
            func_name = line[1]
            func_uuid = line[2]
func_list_fp.close()




def extract_fun_address_ori():
    func_list_arr = []
    func_list_dict = {}
    #chmod_or_fchmod,  3 ,  00x0L ,  '1.70 ,coreutils/coreutils6.7_mipsel32_gcc5.5_o2,coreutils,coreutils6.7
    with open(func_list_ori, "r") as fp:
        for line in csv.reader(fp):
            if line[0] == "":
                continue
            if block_num_max > 0:
                if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                    continue
            # if func_list_dict.has_key(line[0]):
            if line[0]  in func_list_dict:
                value = func_list_dict.pop(line[0])
                value = value + "," + line[3]+"." + line[2]
                func_list_dict.setdefault(line[0],value)
            else:
                #print line
                value = line[3]+"." + line[2]
                func_list_arr.append(line[0])
                func_list_dict.setdefault(line[0],value)
    return  func_list_arr ,func_list_dict



def extract_fun_address():
    func_list_arr = []
    func_list_dict = {}
    #chmod_or_fchmod,  3 ,  00x0L ,  '1.70 ,coreutils/coreutils6.7_mipsel32_gcc5.5_o2,coreutils,coreutils6.7
    with open(func_list_file, "r") as fp:
        for line in csv.reader(fp):
            if line[0] == "":
                continue
            if block_num_max > 0:
                if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                    continue
            # if func_list_dict.has_key(line[0]):
            if line[0]  in func_list_dict:
                value = func_list_dict.pop(line[0])
                value = value + ","  + line[2]
                func_list_dict.setdefault(line[0],value)
            else:
                #print line
                value = line[2]
                func_list_arr.append(line[0])
                func_list_dict.setdefault(line[0],value)
    return func_list_arr, func_list_dict





func_list_arr,func_list_dict=extract_fun_address()
func_list_arr_ori,func_list_dict_ori=extract_fun_address_ori()
random.shuffle(func_list_arr)
random.shuffle(func_list_arr_ori)
func_list_test = func_list_arr



count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
test_fp = open(test_file, "w")
while count < len(func_list_test):
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_test,1)
        #同一个函数名，从已有训练库和制作数据库中取出地址
        value = func_list_dict.get(random_func[0])#地址
        if random_func[0] in func_list_dict_ori:
            value_1 = func_list_dict_ori.get(random_func[0])#版本+地址
            select_list = value_1.split(',')
            selected_list = random.sample(select_list, 1)
            test_fp.write(value + "," + selected_list[0] + ",1\n")
        else:
            continue
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_test,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        test_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0
