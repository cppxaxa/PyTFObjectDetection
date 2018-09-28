import csv

def csv_dict_list(variables_file):
    key = []
    value = []
    
    with open("data.csv", 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            key.append(row[0])
            value.append(row[1])
    
    return key, value

def find_value(name):
    key, value = csv_dict_list('data.csv')
    
    for i in range(len(key)):
        if key[i] == name:
            return value[i]
        
    return None

