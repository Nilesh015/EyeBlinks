import csv
f = open('eyeblink8/11/27122013_154548_cam.tag',"r")
lines = f.readlines()
#lines.index('#end')
for line in lines[lines.index('#start\n') + 1:lines.index('#end')]:
     words = line.split(":")
     with open('data_orig11.csv', 'a', newline='') as myfile: 
          wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
          wr.writerow(words)
     


