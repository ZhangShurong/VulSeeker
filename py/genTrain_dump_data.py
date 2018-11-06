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

dir_folder = "./QQ_bug/bugs/"
ori_dir_folder= "./features/"
train_file = dir_folder + "train"+PREFIX+".csv"
vaild_file = dir_folder + "vaild"+PREFIX+".csv"




def dirlist(path,allfile):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath=os.path.join(path,filename)
        if os.path.isdir(filepath):
            dirlist(filepath,allfile)
            allfile.append(filepath)
    return allfile

bug_result=[]
bug_result=dirlist('./QQ_bug/bugs',bug_result)
#print(bug_result)
all_result=[]
all_result=dirlist('./QQ_bug/QQ9_0_3',all_result)
#print(all_result)

fun_list=bug_result + all_result
#print(fun_list)


# function name, program name, block num, version uid list,
# for file in fun_list:
#     func_list_file = file + "/function_list" + PREFIX + ".csv"
#     print(func_list_file)
#     func_list_fp = open(func_list_file, "w")
#     with open(file + "/block_info.csv", "r") as fp:
#         print (file  + "/block_info.csv")
#         #func_name_set = set()
#         block_num = 0
#         func_name = ""
#         func_uuid = ""
#         for line in csv.reader(fp):
#             #00x0L,chmod_or_fchmod,00x0L,0,1,0,0,0,9,0,0,2,0.0
#
#             if line[1] == func_name:
#                 block_num = block_num + 1
#             else :
#                 print ("new function : ",line[1],line[2])
#                 if not func_name == "" :
#                     #print "             ",k,func_name,func_uuid
#                     #func_name_set.remove(func_name)
#                     func_list_fp.write(func_name.strip()+","+str(block_num)+","+func_uuid.strip()+"\n")
#                 #func_name_set.add(line[1])
#                 block_num = 1
#                 func_name = line[1]
#                 func_uuid = line[2]
#     func_list_fp.close()





def extract_fun_address_all():
    func_list_arr = []
    func_list_dict = {}
    #chmod_or_fchmod,  3 ,  00x0L ,  '1.70 ,coreutils/coreutils6.7_mipsel32_gcc5.5_o2,coreutils,coreutils6.7
    for file in all_result:
        func_list_file = file + "/function_list" + PREFIX + ".csv"
        basename=os.path.basename(file)  #获得文件夹名字
        with open(func_list_file, "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                if block_num_max > 0:
                    if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                        continue
                # if func_list_dict.has_key(line[0]):
                key=basename+'.'+line[0]
                if key in func_list_dict:
                    value = func_list_dict.pop(key)
                    value = value + ","  + basename + "." +line[2]
                    func_list_dict.setdefault(key,value)
                else:
                    #print line
                    value = basename + "." +line[2]
                    func_list_arr.append(key)
                    func_list_dict.setdefault(key,value)
    return func_list_arr, func_list_dict


def extract_fun_address_bug():
    func_list_arr = []
    func_list_dict = {}
    #chmod_or_fchmod,  3 ,  00x0L ,  '1.70 ,coreutils/coreutils6.7_mipsel32_gcc5.5_o2,coreutils,coreutils6.7
    for file in bug_result:
        func_list_file = file + "/function_list" + PREFIX + ".csv"
        basename = os.path.basename(file)
        with open(func_list_file, "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                if block_num_max > 0:
                    if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                        continue
                # if func_list_dict.has_key(line[0]):
                fun_name = basename + '.'+line[0]
                if fun_name  in func_list_dict:
                    value = func_list_dict.pop(fun_name)
                    value = value + ","  + basename + "." +line[2]
                    func_list_dict.setdefault(fun_name,value)
                else:
                    #print line
                    value = basename + "." +line[2]
                    func_list_arr.append(fun_name)
                    func_list_dict.setdefault(fun_name,value)
    return func_list_arr, func_list_dict





func_list_arr,func_list_dict=extract_fun_address_bug()
func_list_arr_ori,func_list_dict_ori=extract_fun_address_all()
random.shuffle(func_list_arr)
random.shuffle(func_list_arr_ori)


def get_test_csv():
    for file in bug_result:
        func_list_test = []
        func_dict_test = {}
        func_list_file = file + "/func_info" + PREFIX + ".csv"
        basename = os.path.basename(file)
        with open(func_list_file, "r") as fp :
            for line in csv.reader(fp):
                bug_key=basename+'.'+line[1].strip()
                test_file = dir_folder + basename + '/'+bug_key + ".csv"
                func_list_test = [bug_key]
                func_dict_test = {bug_key: func_list_dict[bug_key]}
                random_func = random.sample(func_list_test, 1)
                value1 = func_dict_test.get(random_func[0])
                with open(test_file, "w") as test_fp:
                    for key in func_list_arr_ori:
                        print(key)
                        value2 = func_list_dict_ori.get(key)
                        select_list2 = value2.split(',')
                        for selected_list in select_list2:
                            test_fp.write(value1 + "," + selected_list + ",-1\n")


if __name__ =='__main__':
    get_test_csv()

    # count = 0 #记录样本总量
    # cur_num = 0 #记录当前轮次 正例/负例 数量
    # test_fp = open(test_file, "w")
    # func_list_test, func_dict_test = get_func_list_test()
    # # 生成负例
    #
    # for key in func_list_arr_ori:
    #     print(key)
    #     random_func = random.sample(func_list_test,1)
    #     value1 = func_dict_test.get(random_func[0])
    #     value2 = func_list_dict_ori.get(key)
    #     select_list2 = value2.split(',')
    #     for selected_list in select_list2:
    #         test_fp.write(value1+","+selected_list +",-1\n")
