import os
import pandas as pd

def get_nums_pic():
    file = 'WIN5-SAI-ALL.txt'
    info = list()
    with open(file, 'r') as f:
        newno = 0
        for item in f.readlines():
            no, name, score = item.strip().split()
            num = int(name.split('.')[0].split('_')[-1])
            if num % 9 >= 2 and num % 9 <= 8 and num >= 11 and num <= 71:
                newno += 1
                info.append(str(newno) + '\t' + name + '\t' + score + '\n')

    new_file = 'WIN5-SAI-ALL-49.txt'
    with open(new_file, 'w') as f:
        for item in info:
            # print(item)
            f.write(item)

def get_score():
    excel = '../dataset/Win5-LID/Win5-LID_MOS.xlsx'
    file = pd.read_excel(excel)

    res = list()
    for i in range(219):
        name = file['filename'][i]
        score = file['Picture_MOS'][i]
        for j in range(1, 82):
            new_name = name + '_' + str(j) + '.bmp'
            res.append(str(i * 81 + j) + '\t' + new_name + '\t' + str(score) + '\n')

    with open('WIN5-SAI-ALL.txt', 'w')as f:
        for item in res:
            f.write(item)

get_nums_pic()