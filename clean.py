import pandas as pd

ERROR = -1

def L3_shared_temp(x):
    try:
        if(len(x["L3"].split(' ')) > 8):
            return x["L3"].split(' ')[7] + ' ' + x["L3"].split(' ')[8] + ' ' + x["L3"].split(' ')[11]
        else:
            return False
    except:
        return False
    
def trans_L1_I(data):
    try:
        x = data["Cache L1"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try: 
                num = row.split(' ')[0]
                num = float(num)
                r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        print(row)
        return ERROR

def trans_L1_D(data):
    try:
        x = data["Cache L1"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try: 
                num = row.split(' ')[4]
                num = float(num)
                r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        print(row)
        return ERROR

def trans_L2(data):
    try:
        x = data["L2"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try:
                num = row.split(' ')[0]
                type = row.split(' ')[1]
                if (type == 'MB'):
                    num = float(num) * 1024
                else:
                    num = float(num)
                r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        # print(row)
        return ERROR

def trans_L3(data):
    try:
        x = data["L3"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try: 
                num = row.split(' ')[0]
                num = float(num)
                r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        print(row)
        return ERROR

def trans_memory(data):
    try:
        x = data["Memory"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try: 
                num = row.split(' ')[0]
                type = row.split(' ')[1]
                if (type == 'TB'):
                    num = float(num) * 1024
                else:
                    num = float(num)
                r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        print(row)
        return ERROR
    
def trans_storage(data):
    try:
        x = data["Storage"]
        r = pd.Series([])
        for row in x:
            if row == 'redacted':
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                continue
            try: 
                if 'x' in row:
                    num = row.split(' ')[2]
                    type = row.split(' ')[3]
                    if type == 'TB':
                        num = float(num) * 1024
                    else:
                        num = float(num)
                    r = pd.concat([r, pd.Series([num])], ignore_index=True)
                else:
                    num = row.split(' ')[0]
                    type = row.split(' ')[1]
                    if type == 'TB':
                        num = float(num) * 1024
                    else:
                        num = float(num)
                    r = pd.concat([r, pd.Series([num])], ignore_index=True)
            except Exception as e:
                r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
        return r
    except Exception as e:
        print(e)
        print(row)
        return ERROR

csv_file_path = './spider_spec_cfp2017.csv'
data = pd.read_csv(csv_file_path)

# data = data.assign(Cache_L1_I = lambda x:x["Cache L1"].str.split(' ').str[0] + ' ' + x["Cache L1"].str.split(' ').str[1],
#                    Cache_L1_D = lambda x:x["Cache L1"].str.split(' ').str[4] + ' ' + x["Cache L1"].str.split(' ').str[5])
data = data.assign(Cache_L1_I = trans_L1_I, Cache_L1_D = trans_L1_D)
del data["Cache L1"]

# data = data.assign(Cache_L2_ID = lambda x:x["L2"].str.split(' ').str[0] + ' ' + x["L2"].str.split(' ').str[1])
data = data.assign(Cache_L2_ID = trans_L2)
del data["L2"]

# data = data.assign(Cache_L3_ID = lambda x:x["L3"].str.split(' ').str[0] + ' ' + x["L3"].str.split(' ').str[1])
data = data.assign(Cache_L3_ID = trans_L3)
data.loc[:, "L3_shared"] = data.apply(L3_shared_temp, axis=1)
del data["L3"]

data = data.assign(Memory_num = trans_memory)
del data["Memory"]

data = data.assign(Storage_num = trans_storage)
del data["Storage"]

data.to_csv('clean.csv', index=False)
