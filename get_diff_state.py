import math

def floats_are_close(f1, f2, tol=1e-5):
    return math.isclose(f1, f2, abs_tol=tol)

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    mismatches = []
    max_lines = min(len(lines1), len(lines2))

    for i in range(max_lines):
        numbers1 = list(map(float, lines1[i].strip().split()))
        numbers2 = list(map(float, lines2[i].strip().split()))

        if len(numbers1) != len(numbers2):
            mismatches.append((i, '列数不同'))
        else:
            for j in range(len(numbers1)):
                if not floats_are_close(numbers1[j], numbers2[j]):
                    mismatches.append((i, j))

    if len(lines1) != len(lines2):
        mismatches.append(('行数不同',))

    return mismatches

file1 = '/home/uisee/Documents/my_script/offlineFMDQN/data/state_c.txt'
file2 = '/home/uisee/Documents/my_script/offlineFMDQN/data/state_py.txt'

diff = compare_files(file1, file2)
if diff:
    for mismatch in diff:
        print(f"不匹配发生在: {mismatch}")
else:
    print("两个文件在五位小数精度内完全相同。")