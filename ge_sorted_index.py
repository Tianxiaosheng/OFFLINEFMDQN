import numpy as np

def read_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            # 移除行尾的逗号并分割字符串
            numbers = line.strip(',\n').split(',')
            # 将字符串转换为浮点数
            data.append([float(num) for num in numbers])
    return np.array(data)

def sort_indices_desc(data):
    # 对每行数据进行排序，并返回递减排序的索引
    sorted_indices = [np.argsort(-row) for row in data]
    return np.array(sorted_indices)

def write_sorted_indices(filepath, sorted_indices):
    with open(filepath, 'w') as file:
        for indices in sorted_indices:
            # 将索引转换为字符串，并用逗号连接
            line = ','.join(map(str, indices))
            file.write(line + '\n')

def main():
    # 读取数据
    data_a = read_data('data/state_action_values_c.txt')
    data_b = read_data('data/state_action_values_py.txt')

    # 获取排序后的索引
    sorted_indices_a = sort_indices_desc(data_a)
    sorted_indices_b = sort_indices_desc(data_b)

    # 写入新的文件
    write_sorted_indices('data/sorted_indices_c.txt', sorted_indices_a)
    write_sorted_indices('data/sorted_indices_py.txt', sorted_indices_b)

    print("Sorted indices have been written to the new files.")

if __name__ == "__main__":
    main()