import os

path = '/Users/mianmaokuchuanma/database/win5-lid/Win5-LID/EPIs/'

files = os.listdir(path)

with open('WIN5-LID-real.txt') as f:
    allinfo = f.readlines()

res = list()
idx = 0
for item in allinfo:
    _, name, score = item.split('\t')
    score = score.split('\n')[0]
    if name in files:
        res.append(str(idx) + '\t' + name + '\t' + str(score) + '\n')
        idx += 1

with open('WIN5-EPI.txt', 'w') as f:
    for item in res:
        f.write(item)



