# click_data_function
import json
import pandas as pd
def de_click_data(filepath=None):
    # data_25 = pd.read_csv('data_0608/test_buy_25.csv',names=['item_id','role_id','num'],delimiter='\t')
    # item_click = list(data_25['item_id'].unique())
    data_click = pd.read_csv('data_0619/click_all_0620.csv',delimiter='\t',names=['role_id','item_list','dt'])
    data_click = data_click[data_click['item_list']!='{}']
    data_click = data_click.reset_index(drop=True)
    data_click['item_list_'] = data_click['item_list'].apply(lambda x:x[1:-1])
    data_ret = pd.DataFrame()
    fin = data_click.shape[0]
    print fin

    for i in range(2000):
        print i
        first = i*10000
        last = (i+1)*10000-1
        if (i+1)*10000 > fin:
            last = fin
        temp = data_click.ix[first:last]['item_list_'].str.split(',',expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'item_list_'}).join(data_click.ix[first:last].drop(['item_list_','item_list'], axis=1))
        # temp = pd.concat(temp,temp['item_list_'].str.split(':',expand=True).rename(columns ={0:'item_id',1:'cnt'}))
        # pd.concat([temp,temp.item_list_.str.split(':',1).rename(columns ={0:'item_id',1:'cnt'})],axis=1)
        temp[["item_id","cnt"]] = temp["item_list_"].apply(lambda x: pd.Series([i for i in x.split(":")]))
        # temp = temp[temp['item_id'].apply(lambda x: int(x) in item_click)]
        data_ret = pd.concat([data_ret,temp.drop('item_list_',axis=1)],axis=0)
        if (i+1)*10000 > fin:
            break
        data_ret = data_ret.reset_index(drop=True)

    data_ret = data_ret.reset_index(drop=True)
    data_ret.to_csv('data_0619/data_click_item_all.csv')


def split_gender():
    for i in range(19,22):
        print i
        data = pd.read_csv('data_0619/src_appearance_0621_05{}.csv'.format(str(i)),delimiter='\t',names=['server','role_id','wear','dt'])
        def funcc(x):
            k = eval(x)
            if k!=[]:
                return k.get('Gender')
            else:
                return None
        def funcc2(x):
            k = eval(x)
            if k!=[]:
                return k.get('Tshirt')
            else:
                return None
        def funcc3(x):
            k = eval(x)
            if k!=[]:
                return k.get('Pants')
            else:
                return None
        def funcc4(x):
            k = eval(x)
            if k!=[]:
                return k.get('Hat')
            else:
                return None
        def funcc5(x):
            k = eval(x)
            if k!=[]:
                return k.get('Hair')
            else:
                return None

        data['gender'] = data['wear'].apply(funcc)
        data['tshirt'] = data['wear'].apply(funcc2)
        data['pants'] = data['wear'].apply(funcc3)
        data['hat'] = data['wear'].apply(funcc4)
        data['hair'] = data['wear'].apply(funcc5)
        data.to_csv('data_0619/src_appearance_0621_05{}_split.csv'.format(str(i)))

# de_click_data()
if __name__ == '__main__':
    from pyfm import pylibfm
    from sklearn.feature_extraction import DictVectorizer
    import numpy as np

    train = [
        {"user": "1", "item": "5", "age": 19},
        {"user": "2", "item": "43", "age": 33},
        {"user": "3", "item": "20", "age": 55},
        {"user": "4", "item": "10", "age": 20},
    ]
    v = DictVectorizer()
    X = v.fit_transform(train)
    print(X.toarray())

    y = np.repeat(1.0, X.shape[0])
    fm = pylibfm.FM()
    fm.fit(X, y)
    fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
