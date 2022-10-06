import os
import pandas as pd
import math
import random

def get_nums_pic():
    file = 'NEW_WIN5_SAI.txt'
    info = list()
    with open(file, 'r') as f:
        newno = 0
        for item in f.readlines():
            no, name, score = item.strip().split()
            num = int(name.split('.')[0].split('_')[-1])
            if num % 9 >= 2 and num % 9 <= 8 and num >= 11 and num <= 71:
                newno += 1
                info.append(str(newno) + '\t' + name + '\t' + score + '\n')

    new_file = 'NEW_WIN5-SAI-49.txt'
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
        score = file['Overall_MOS'][i]
        for j in range(1, 82):
            new_name = name + '_' + str(j) + '.bmp'
            res.append(str(i * 81 + j) + '\t' + new_name + '\t' + str(score) + '\n')

    with open('WIN5-SAI-ALL.txt', 'w')as f:
        for item in res:
            f.write(item)

def get_pics():
    path = '../dataset/Win5-LID/Distorted/Real'
    excel = '../dataset/Win5-LID/Win5-LID_MOS.xlsx'
    excel_file = pd.read_excel(excel)
    pics = os.listdir(path)

    tmp = dict()
    for i in range(220):
        name = excel_file['filename'][i]
        score = excel_file['Overall_MOS'][i]
        tmp[name] = score

    res = list()
    for i in range(len(pics)):
        name = pics[i].split('.')[0]
        # print(name)
        score = tmp[name]
        # for j in range(1, 82):
        #     new_name = name + '_' + str(j) + '.bmp'
        res.append(str(i + 1) + '\t' + name + '\t' + str(score) + '\n')

    with open('WIN5-LFI-Real.txt', 'w') as f:
        for item in res:
            f.write(item)

def pre_pic_score():
    file = 'WIN5-LID-real.txt'
    with open(file, 'r') as f:
        info = f.readlines()

    res = list()
    for i in range(len(info)):
        no, name, score = info[i].split()
        strn = name.split('.')[0][-2:]
        if(strn[0] == '_'):
            num = int(strn[1:])
        else:
            num = int(int(strn[0]) * 10 + int(strn[1]))
        if num % 9:
            line = num / 9 + 1
            row = num % 9
        else:
            line = num / 9
            row = 9
        d = math.sqrt((line - 5) ** 2 + (row - 5) ** 2) * 1e-6
        score = float(score) - d
        write_to = no + '\t' + name + '\t' + str(score) + '\n'

        res.append(write_to)

    with open('NEW_WIN5_SAI.txt', 'w')as f:
        for item in res:
            f.write(item)

def split_train_test():
    raw_txt = 'WIN5-LFI-Real.txt'
    with open(raw_txt, 'r') as f:
        info = f.readlines()


    train_samples = random.sample(info, 106)
    test_samples = []
    for item in info:
        if item not in train_samples:
            test_samples.append(item)

    print('train', len(train_samples))
    print('test', len(test_samples))

    res = list()
    num = 0
    for item in train_samples:
        no, name, score = item.split()
        for i in range(11, 72):
            if i % 9 >= 2 and i % 9 <= 8:
                new_name = name + '_' + str(i) + '.bmp'
                write_to = str(num) + '\t' + new_name + '\t' + score + '\n'
                num = num + 1
                res.append(write_to)

    train_txt = 'WIN5-SAI-49-train-2.txt'
    with open(train_txt, 'w') as f:
        for item in res:
            f.write(item)

    res = []
    num = 0
    for item in test_samples:
        no, name, score = item.split()
        for i in range(11, 72):
            if i % 9 >= 2 and i % 9 <= 8:
                new_name = name + '_' + str(i) + '.bmp'
                write_to = str(num) + '\t' + new_name + '\t' + score + '\n'
                num = num + 1
                res.append(write_to)

    test_txt = 'WIN5-SAI-49-test-2.txt'
    with open(test_txt, 'w') as f:
        for item in res:
            f.write(item)


def split_train_test2():
    raw_txt = 'WIN5-LFI-Real.txt'
    with open(raw_txt, 'r') as f:
        info = f.readlines()
    pic_dict = {
        'EPICNN':[],
        'HEVC':[],
        'NN':[],
        'LN':[],
        'USCD':[],
        'JPEG':[]
    }

    train_samples = []
    test_samples = []
    for item in info:
        name = item.split()[1]
        score = item.split()[2]
        if 'EPICNN' in name:
            pic_dict['EPICNN'].append(name+ '\t' + score)
        elif 'HEVC' in name:
            pic_dict['HEVC'].append(name+ '\t' + score)
        elif 'NN' in name:
            pic_dict['NN'].append(name+ '\t' + score)
        elif 'LN' in name:
            pic_dict['LN'].append(name+ '\t' + score)
        elif 'USCD' in name:
            pic_dict['USCD'].append(name+ '\t' + score)
        elif 'JPEG' in name:
            pic_dict['JPEG'].append(name+ '\t' + score)

    for key in pic_dict.keys():
        if key == 'EPICNN':
            train_samples.extend(pic_dict['EPICNN'][1:])
            test_samples.append(pic_dict['EPICNN'][0])
        elif key == 'USCD':
            train_samples.extend(pic_dict['USCD'][2:])
            test_samples.extend(pic_dict['USCD'][:2])
        else:
            train_lst = random.sample(pic_dict[key], 24)
            for pic in pic_dict[key]:
                if pic in train_lst:
                    train_samples.append(pic)
                else:
                    test_samples.append(pic)
    print('train', len(train_samples))
    print('test', len(test_samples))

    res = list()
    num = 0
    for item in train_samples:
        name = item.split()[0]
        score = item.split()[1]
        for i in range(11, 72):
            if i % 9 >= 2 and i % 9 <= 8:
                new_name = name + '_' + str(i) + '.bmp'
                write_to = str(num) + '\t' + new_name + '\t' + score + '\n'
                num = num + 1
                res.append(write_to)

    train_txt = 'WIN5-SAI-49-new-train-1.txt'
    with open(train_txt, 'w') as f:
        for item in res:
            f.write(item)

    res = []
    num = 0
    for item in test_samples:
        name = item.split()[0]
        score = item.split()[1]
        for i in range(11, 72):
            if i % 9 >= 2 and i % 9 <= 8:
                new_name = name + '_' + str(i) + '.bmp'
                write_to = str(num) + '\t' + new_name + '\t' + score + '\n'
                num = num + 1
                res.append(write_to)

    test_txt = 'WIN5-SAI-49-new-test-1.txt'
    with open(test_txt, 'w') as f:
        for item in res:
            f.write(item)

split_train_test2()
