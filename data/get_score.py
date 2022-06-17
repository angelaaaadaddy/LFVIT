import os
import xlrd
import pandas as pd


def get_pic_names():
    LFI_path = '/Users/mianmaokuchuanma/DATABASE/Win5-LID/Win5-LID/Distorted/Real'
    files = os.listdir(LFI_path)

    pic_names = []
    for item in files:
        if not item.endswith('bmp'):
            continue
        pic_names.append(item[:-4])

    return pic_names

def get_excel_info(pic_list):
    file_path = '/Users/mianmaokuchuanma/DATABASE/Win5-LID/Win5-LID/Win5-LID_MOS.xlsx'
    df = pd.read_excel(file_path, sheet_name=[0])
    # print(df.values()['Picture_MOS'])
    datas = list()
    for k, v in df.items():
        for index, row in v.iterrows():
            datas.append([row['filename'], float(row['Picture_MOS'])])

    otpt = list()
    for item in datas:
        if item[0] in pic_list:
            otpt.append(item)

    return otpt

pic_list = get_pic_names()
datas = get_excel_info(pic_list)
# print(datas)

with open('../IQA_list/WIN5-LFI.txt', 'w') as f:
    for i in range(len(datas)):
        write_to = str(i) + '\t' + datas[i][0] + '.bmp' + '\t' + str(datas[i][1]) + '\n'
        f.write(write_to)