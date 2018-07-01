import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing

def save_model(filename, data):
    pickle.dump(data, open('data_0619/' + filename + '.pickle', 'wb'), protocol=2)

def load_model(filename):
    return pickle.load(open('data_0619/' + filename + '.pickle', 'rb'))
# def save_

class onehotcode(object):
    def __init__(self, item_id):
        self.enc = preprocessing.OneHotEncoder()
        self.enc.fit(item_id)
    # item_feature = pd.concat([item_feature, pd.DataFrame(enc.fit_transform(np.array(item_feature['item_id']).reshape(-1, 1)).toarray())], axis=1)

    def OnehotCode(self, item_id):
        return self.enc.transform(item_id)

class predict_info(object):
    def __init__(self):
        self.cor_1,self.cor_3,self.cor_5,self.cor_10,self.cor_15,self.cor_20,self.cor_25,self.tol = [0,0,0,0,0,0,0,0]
        self.pri_1,self.pri_3,self.pri_5,self.pri_10,self.pri_15,self.pri_20,self.pri_25 = [0,0,0,0,0,0,0]

    def restart(self):
        self.cor_1, self.cor_3, self.cor_5, self.cor_10, self.cor_15, self.cor_20, self.cor_25, self.tol = [0, 0, 0, 0,
                                                                                                            0, 0, 0, 0]
        self.pri_1, self.pri_3, self.pri_5, self.pri_10, self.pri_15, self.pri_20, self.pri_25 = [0, 0, 0, 0, 0, 0, 0]

    def add(self, k, price):
        if k < 25:
            self.cor_25 += 1
            self.pri_25 += price
        if k < 20:
            self.cor_20 += 1
            self.pri_20 += price
        if k < 15:
            self.cor_15 += 1
            self.pri_15 += price
        if k < 10:
            self.cor_10 += 1
            self.pri_10 += price
        if k < 5:
            self.cor_5 += 1
            self.pri_5 += price
        if k < 3:
            self.cor_3 += 1
            self.pri_3 += price
        if k < 1:
            self.cor_1 += 1
            self.pri_1 += price
        self.tol += 1

    def print_result(self):
        print 'over, the answei is -------------------------'
        print 'pre_1 is', self.cor_1 * 100.0 / self.tol, '%', 'price is', self.pri_1
        print 'pre_3 is', self.cor_3 * 100.0 / self.tol, '%', 'price is', self.pri_3
        print 'pre_5 is', self.cor_5 * 100.0 / self.tol, '%', 'price is', self.pri_5
        print 'pre_10 is', self.cor_10 * 100.0 / self.tol, '%', 'price is', self.pri_10
        print 'pre_15 is', self.cor_15 * 100.0 / self.tol, '%', 'price is', self.pri_15
        print 'pre_20 is', self.cor_20 * 100.0 / self.tol, '%', 'price is', self.pri_20
        print 'pre_25 is', self.cor_25 * 100.0 / self.tol, '%', 'price is', self.pri_25

def rec_pre(pre, item_id, pre_default):
    pre = pd.merge(pre, item_id, on='item_id', how='outer').fillna(0.05)
    return list(pre.sort_values(by=['preb'], ascending=False).reset_index(drop=True)['item_id'])

def get_newitem_list(dt):
    new_item = pd.read_csv('data_0608/new_item.csv', names=['item_id', 'item_name', 'newdt', 'dt'], delimiter='\t')
    new_item = new_item[new_item['newdt'] > (dt - 7)]
    return list(new_item)

if __name__ == '__main__':
    gender_enc = onehotcode(np.array([1, 2, 3]).reshape(-1, 1))
    print gender_enc.OnehotCode(np.array([1,2,3]).reshape(-1, 1)).toarray()