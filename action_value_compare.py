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

def calculate_mse(data1, data2):
    # 计算MSE
    return np.mean((data1 - data2) ** 2)

def main():

    # 读取数据
    data_a = read_data('data/state_action_values_c.txt')
    data_b = read_data('data/state_action_values_py.txt')
    # data_a = read_data('data/a.txt')
    # data_b = read_data('data/b.txt')

    # 计算MSE
    mse = calculate_mse(data_a, data_b)
    print(f"MSE: {mse}")

if __name__ == "__main__":
    main()