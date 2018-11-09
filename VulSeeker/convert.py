def load_data(csv_path):
    lines = []
    with open(csv_path) as f:
        lines = f.readlines()
    data_dict = dict()
    for line in lines:
        data_arr = line.split(',')
        probability = float(data_arr[0])
        cve_func = data_arr[1]
        search_fun = data_arr[2]
        if search_fun not in data_dict:
            data_dict[search_fun] = []
        found = False
        for i in data_dict[search_fun]:
            if i[0] == cve_func:
                found = True
        if not found:
            item_dict = (cve_func, probability)
            data_dict[search_fun].append(item_dict)
    return data_dict

def save_csv(data_dict, csv_path):
    lines = []
    with open(csv_path, 'w') as output:
        for key, data_arr in data_dict.iteritems():
            line = key.strip()
            pro = 0.0
            count = 0.0
            for data in data_arr:
                line = line + ',' + str(data[1])
                pro += data[1]
                count += 1
            line += str(pro/count)
            line += '\n'
            lines.append(line)
        output.writelines(lines)
def main():
    # 1. load ori data
    res = load_data('vulseeker.csv')
    # 2. save data to a new csv file
    save_csv(res, 'New_vulseeker.csv')

if __name__ == '__main__':
    main()
