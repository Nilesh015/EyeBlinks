import pandas as pd
import numpy as np
import math
import random

import csv

with open('data_orig11.csv', 'r') as f:
    reader = csv.reader(f)
    y1 = list(reader)

y = np.asarray(y1)
print(len(y1))

data = pd.read_csv('EAR_orig_11.csv',header = 0)
data = data.dropna()

data_vars=data.columns.values.tolist()
data_final = data[data_vars]

ear = data_final.loc[:, data_final.columns == 'EAR']
ear = np.asarray(ear)
print(ear.shape[0])

conv = []
p1=0
lo = 0
while (lo-p1) < (y.shape[0]):
    if(int(y[lo-p1][0]) != lo):
    	p1 += 1
    	lo += 1
    	continue
    conv.append([ y[lo-p1][0], y[lo-p1][1], ear[lo][0] ])
    lo += 1
conv = np.asarray(conv)
print(conv)

count = 0
i = 0

while i < (conv.shape[0]):

    if (conv[i][1] == '-1'):
        conv[i][1] = '0'

    elif(conv[i][1] != '-1' and conv[i][1] != '0'):
        conv[i][1] = '1'
    i += 1

def group1(y):
    c=0
    i=0
    y_final = []
    while i < (conv.shape[0]-7):
        c=0
        while(conv[i+c][1] == '1'):
              c += 1
              if( i+c == conv.shape[0]):
                  i += c + 1
                  c = 0
                  break
        if c > 0:
            if c > 10:
                i += c
                continue
            x = []
            for k in range((i + int(c/ 2) - 6), (i + int(c/2) + 7)):
                x.append(conv[k])
            x = np.asarray(x)
            c1 = 0
            y_temp = []
            y_temp.append('1')
            print(i + int(c/2) + 7)
            i += c
            while c1 < (x.shape[0]):
                y_temp.append(x[c1][2])
                c1 += 1
            y_final.append(y_temp)
            print(str(len(y_final)) + "," + str(i))

        else:
            i += 1
    return y_final

def group0(y,n):
    c=0
    i=0
    p=1
    y_final = []
    while p <= n:
        i = random.randint(0, int(conv.shape[0]/n) * p)
        c=0
        if(conv[i][1] == '1'):
              c += 1
              
        if c > 0:
            continue
        else:
            if c == 0:
                while (conv[i + c][1] == '0' and c < 13):
                    c += 1
            if(c < 13):
                continue
            x = []
            for k in range((i + int(c/ 2) - 6), (i + int(c/2) + 7)):
                x.append(conv[k])
            x = np.asarray(x)
            c1 = 0
            p += 1
            y_temp = []
            y_temp.append('0')
            while c1 < (x.shape[0]):
                y_temp.append(x[c1][2])
                c1 += 1
            y_final.append(y_temp)
            print(str(len(y_final)) + "," + str(i))

    return y_final

Y_final1 = group1(y)
Y_final0 = group0(y,int(len(Y_final1)))

for m in range(0,len(Y_final0)):
    Y_final1.append(Y_final0[m])

Y_final1 = np.asarray(Y_final1)

print(Y_final1.shape)

#with open('testEAR.csv', 'a', newline='') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(['y', 'EAR_6', 'EAR_5', 'EAR_4', 'EAR_3', 'EAR_2', 'EAR_1', 'EAR0', 'EAR1', 'EAR2', 'EAR3', 'EAR4', 'EAR5', 'EAR6'])

for i in range(Y_final1.shape[0]):
    with open('trainEAR_FULL.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(Y_final1[i])

