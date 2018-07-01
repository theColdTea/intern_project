# coding:utf8

import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import copy
import gc
class recommendation_fm(object):
    def __init__(self):
        new_item = pd.read_csv('data_0608/new_item.csv', names=['item_id', 'item_name', 'newdt', 'dt'], delimiter='\t')
        # self.new_item_list = list(new_item[new_item['newdt']>20180430]['item_id'])
        item_id = pd.read_csv('data_0608/item_id.csv', header=None)
        item_id = list(np.array(item_id[0]))
        self.item_id = [x for x in item_id if x not in [991, 992, 993, 994, 995]]

        # self.new_item_list = list(new_item[new_item['newdt']>20180520]['item_id'])
        self.cor_1 = 0
        self.cor_3 = 0
        self.cor_5 = 0
        self.cor_10 = 0
        self.cor_15 = 0
        self.cor_20 = 0
        self.cor_25 = 0

    def train_data(self, filepath=None,data=None):
        if filepath is not None:
            data = pd.read_csv(filepath, index_col=0)
        else:
            data = data
        gc.collect()
        # 1 10000 9 9013 9023 9015 9005 9019 1022 1018 9009 9011 9007( delete the number of item > 10000
        # 去掉之后很多用户没有信息了
        ex_list = [1,10000,9,9001,9019,1020,1018,1014,0,2,45]
        # ex_list =
        data = data[data['item_id'].apply(lambda x: x not in ex_list)]
        # normalize
        data = pd.merge(data, data.groupby('item_id')['num'].sum().reset_index(), on='item_id')
        # data = pd.merge(data, data.groupby('item_id')['num'].median().reset_index(), on='item_id')
        data['num_s'] = data.assign(num_s=lambda x: x.num_x >= x.num_y)['num_s'].apply(lambda x: 1 if x else 0)
        # data['num_s'] = data['num'].apply(lambda x: (x - 1) * 1.0 / 9 + 1 if x > 1 else x)
        # encode item_id and role_id
        # uid : 0,1,2... --> role_id
        self.uid = dict(zip(pd.DataFrame(data['role_id'].unique()).reset_index()['index'],pd.DataFrame(data['role_id'].unique()).reset_index()[0]))
        self.re_uid = dict(zip(pd.DataFrame(data['role_id'].unique()).reset_index()[0],pd.DataFrame(data['role_id'].unique()).reset_index()['index']))

        self.iid = dict(zip(pd.DataFrame(data['item_id'].unique()).reset_index()['index'],pd.DataFrame(data['item_id'].unique()).reset_index()[0]))
        self.re_iid = dict(zip(pd.DataFrame(data['item_id'].unique()).reset_index()[0],pd.DataFrame(data['item_id'].unique()).reset_index()['index']))
        data['role_id_index'] = data['role_id'].apply(lambda x: self.re_uid[x])
        data['item_id_index'] = data['item_id'].apply(lambda x: self.re_iid[x])
        return data

    def test_data(self, filepath):
        data = pd.read_csv(filepath, names=['item_id', 'role_id', 'num'], delimiter='\t')
        return data

    def train(self, train, n_factor, epochs, lr, loss):
        train = sparse.coo_matrix((train['num_s'], (train['role_id_index'], train['item_id_index'])))

        model = LightFM(no_components=n_factor, learning_rate=lr, loss=loss)
        model.fit(train, epochs=epochs, verbose=True)
        return model

    def predict(self, model, test_filepath, train):
        test = self.test_data(test_filepath)
        self.cor_cof_clear()
        test_role_id = test['role_id'].unique()
        pre_popu = []
        a = 0
        for i in list(train['item_id'].value_counts().reset_index()['index']):
            if i in self.item_id:
                pre_popu.append(i)

        for role in test_role_id:
            try:
                role_index = self.re_uid[role]
                pre = model.predict(role_index, np.arange(train.shape[1]))
                pre = [self.iid[x] for x in np.argsort(-pre) if x in self.item_id]

            except:
                a+=1
                print 'the role_id is', role, 'is the first time buying...'
                pre = copy.copy(pre_popu)

            # for k in self.new_item_list:
            #     try: pre.remove(k)
            #     except: continue

            # for k in list(train[train['role_id'] == role]['item_id']):
            #     try: pre.remove(k)
            #     except: continue

            item_real = test[test['role_id'] == role]['item_id'].values
            for i in item_real:
                # if i in self.new_item_list:
                #     continue
                try:
                    self.add_cor(pre.index(i))
                    print i, pre.index(i)
                except:
                    print i, 'the role', role, 'has buy twice'
                # print i, pre.index(i)

        print 'over, the answei is -------------------------'
        print a,'roles id the firstime'
        print 'pre_1 is', cor_1 * 100 / tol, '%'
        print 'pre_3 is', cor_3 * 100 / tol, '%'
        print 'pre_5 is', cor_5 * 100 / tol, '%'
        print 'pre_10 is', cor_10 * 100 / tol, '%'
        print 'pre_15 is', cor_15 * 100 / tol, '%'
        print 'pre_20 is', cor_20 * 100 / tol, '%'
        print 'pre_15 is', cor_25 * 100 / tol, '%'

    def add_cor(self, k):
        global cor_1, cor_3, cor_5, cor_10, cor_15, cor_20, cor_25, tol
        if k < 25: cor_25 += 1
        if k < 20: cor_20 += 1
        if k < 15: cor_15 += 1
        if k < 10: cor_10 += 1
        if k < 5:  cor_5 += 1
        if k < 3:  cor_3 += 1
        if k < 1:  cor_1 += 1
        tol+=1

    def cor_cof_clear(self):
        global cor_1, cor_3, cor_5, cor_10, cor_15, cor_20, cor_25, tol
        cor_1 = 0
        cor_3 = 0
        cor_5 = 0
        cor_10 = 0
        cor_15 = 0
        cor_20 = 0
        cor_25 = 0
        tol = 0

    def coe_of_items(self, item_list_1, item_list_2, train):
        for item1 in item_list_1:
            for item2 in item_list_2:
                print item2
                data_temp1 = train.copy()
                data_temp1 = data_temp1.groupby(['role_id', 'item_id']).sum()
                data_temp1 = data_temp1.fillna(0).reset_index()
                data_temp1 = pd.merge(data_temp1[data_temp1['item_id'] == item1],
                                      data_temp1[data_temp1['item_id'] == item2], on=['role_id'], how='outer')
                print item1, 'with', item2
                print data_temp1
                co = data_temp1[['num_x', 'num_y']].corr().ix['num_x', 'num_y']
                #         if co > 0.03:
                print item1, item2, co



def add_fm_vec(train, user_feature, item_feature, dt, user_enable=True, item_enable=True):
    rec = recommendation_fm()
    click_data = pd.read_csv('data_0619/data_click_item_all.csv',index_col=0)
    click_data = click_data[(click_data['dt']<dt) & (click_data['dt']>dt-8)].rename(columns={'cnt':'num'})
    # train_rec = rec.train_data('data_0619/buyall_{}.csv'.format(str(dt-1)[-4:]))
    train_rec = rec.train_data(data=click_data)

    fm_model = rec.train(train_rec, n_factor=10, epochs=10, lr=0.01, loss='warp')
    print 'fm_train is over'
    if user_enable:
        user_feature_fm = pd.DataFrame(fm_model.user_embeddings)
        user_feature_fm.rename(columns=lambda x: str(x) + '_user_fm', inplace=True)
        user_feature_fm = user_feature_fm.reset_index().rename(columns={'index': 'role_id'})
        user_feature_fm['role_id'] = user_feature_fm['role_id'].apply(lambda x: rec.uid[x])

        user_feature = pd.merge(user_feature, user_feature_fm, on='role_id', how='outer').fillna(0)
    if item_enable:
        item_feature_fm = pd.DataFrame(fm_model.item_embeddings)
        item_feature_fm.rename(columns=lambda x: str(x) + '_item_fm', inplace=True)
        item_feature_fm = item_feature_fm.reset_index().rename(columns={'index': 'item_id'})
        item_feature_fm['item_id'] = item_feature_fm['item_id'].apply(lambda x: rec.iid[x])

        item_feature = pd.merge(item_feature, item_feature_fm, on='item_id', how='outer').fillna(0)

    def funcc(a, b):
        try:
            return fm_model.predict(rec.re_uid[a], [rec.re_iid[b]])[0]
        except:
            return -2
    train['fm'] = train.apply(lambda row: funcc(row['role_id'], row['item_id']), axis=1)

    return train,user_feature, item_feature