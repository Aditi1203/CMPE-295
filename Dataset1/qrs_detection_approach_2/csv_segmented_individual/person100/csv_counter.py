import csv
with open('6.csv') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='|')
    count = 0
    for row in spamreader:
        print(len(row))
    # print(len(row_list))