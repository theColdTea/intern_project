# coding:utf8

import pandas as pd
import numpy as np

from scipy import sparse
import copy

from recommedation_fm import recommendation_fm, add_fm_vec
from recommend_util import save_model, onehotcode, load_model,predict_info, rec_pre, get_newitem_list

def lgb_train(data):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    '''
    import xlearn as xl
    temp = list(data.columns)
    temp.remove('label')
    temp = ['label'] + temp
    data[temp].to_csv('train_temp_fm.csv',header=None,index=0)
    fm_model = xl.create_fm()
    fm_model.setTrain('train_temp_fm.csv')
    param = {'task': 'binary', 'lr': 0.2,
             'lambda': 0.002, 'metric': 'acc'}
    fm_model.cv(param)
    fm_model.predict('out.model','out.csv')
    '''

    params = {'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.02,
        'max_depth': 4,
        'num_leaves': 15,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.6,
        'bagging_freq': 17
             }

    train, val = train_test_split(data, test_size=0.2, random_state=21)
    y_train = train.label  # 训练集标签
    X_train = train.drop(['label'], axis=1)  # 训练集特征矩阵

    y_val = val.label  # 验证集标签
    X_val = val.drop(['label'], axis=1)  # 训练集特征矩阵

    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)
    print 'cross validation'
    best_params = {}
    print 'adjust params 1st'

    # for num_leaves in range(20, 200, 5):
    #     for max_depth in range(3, 8, 1):
    #         params['num_leaves'] = num_leaves
    #         params['max_depth'] = max_depth
    #
    #         cv_results = lgb.cv(
    #             params,
    #             lgb_train,
    #             seed=2018,
    #             nfold=3,
    #             metrics=['binary_error'],
    #             early_stopping_rounds=10,
    #         )
    #

    model = lgb.train(
        params,  # 参数字典
        lgb_train,  # 训练集
        valid_sets=lgb_eval,  # 验证集
        num_boost_round=10000,  # 迭代次数
        early_stopping_rounds=20,  # 早停次数
    )

    return model

def gen_pos_sample_index(dt):
    # all date
    data = pd.read_csv('data_0608/src_mall_day_0615.csv',delimiter='\t',header=None,names=['role_id','item_id','buy_num','dt'])
    data = data[data['dt']==dt]
    print 'pos', data.shape
    data.drop_duplicates(['item_id','role_id'])
    return data[['item_id','role_id']]

def gen_click_feature(dt, active_role, item_id, user_feature, item_feature,train_index=None,sample_rate=None):
    # all date
    print 'item_shape',item_id.shape
    data = pd.read_csv('data_0619/data_click_item_all.csv',index_col=0)
    role_data = data[data['dt']==(dt-1)][['role_id','item_id']]
    data = data[data['dt']<dt]
    data = pd.merge(active_role, data, on='role_id', how='inner')
    data = pd.merge(item_id, data, on='item_id', how='inner')

    # if train_index is not None:
    #
    #     # 随机补齐一些负样本，随机定的0.2
    #     role_user = train_index['role_id']
    #     role_temp = data['role_id'].drop_duplicates().sample(frac=0.2).reset_index().drop('index', axis=1)
    #     item_temp = item_id.copy()
    #     role_temp['key'] = 0
    #     item_temp['key'] = 0
    #     new_data = pd.merge(role_temp, item_temp, on='key').drop('key', axis=1)
    #     new_data['cnt'] = 0
    #     data = pd.merge(data, new_data, on=['role_id', 'item_id'], how='outer').fillna(0)
    #     data['cnt'] = data['cnt_x'] + data['cnt_y']
    #     data = data.drop(['cnt_x', 'cnt_y'], axis=1)
    #     data['dt'] = data['dt'].replace(0, dt-10)
    #
    #     for role in data['role_id'].sample(frac=0.2).values:
    #         item_id_temp = item_id.copy()
    #         item_id_ex = data[data['role_id']==role]['item_id'].drop_duplicates().reset_index()
    #         item_id_temp = pd.merge(item_id_temp, item_id_ex, on='item_id', how='left')
    #         item_id_temp = item_id_temp[item_id_temp['index'].isnull()].drop('index', axis=1)
    #         item_id_temp['dt'] = dt-14
    #         item_id_temp['role_id'] = role
    #         item_id_temp['cnt'] = 0
    #         data = pd.concat([data, item_id_temp])
        

    # if train_index is not None:
    #     # 随机补齐一些负样本，随机定的0.3
    #     # random_user = data['role_id'].nunqiue()*0.2
    #     role_temp = data['role_id'].drop_duplicates().sample(frac=0.05).reset_index().drop('index', axis=1)
    # # else:
    # #     role_temp = data['role_id'].drop_duplicates().reset_index().drop('index', axis=1)
    #     item_temp = item_id.copy()
    #     role_temp['key'] = 0
    #     item_temp['key'] = 0
    #     new_data = pd.merge(role_temp, item_temp, on='key').drop('key', axis=1)
    #     new_data['cnt'] = 0
    #     data = pd.merge(data, new_data, on=['role_id', 'item_id'], how='outer').fillna(0)
    #     data['cnt'] = data['cnt_x'] + data['cnt_y']
    #     data = data.drop(['cnt_x', 'cnt_y'], axis=1)
    #     data['dt'] = data['dt'].replace(0, dt - 22)
    data['d_dt_click'] = dt - data['dt'].astype(int)
    data = data.reset_index(drop=True)

    data = data.drop('dt', axis=1).groupby(['role_id', 'item_id', 'd_dt_click']).sum().reset_index()
    data = pd.merge(data, data[data['d_dt_click'] <= 1].groupby(['role_id', 'item_id'])['cnt'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_1']).fillna(0)
    data = pd.merge(data, data[data['d_dt_click'] <= 3].groupby(['role_id', 'item_id'])['cnt'].sum().reset_index()
                    , on=['role_id', 'item_id'], how='left', suffixes=['', '_3']).fillna(0)
    data = pd.merge(data, data[data['d_dt_click'] <= 7].groupby(['role_id', 'item_id'])['cnt'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_7']).fillna(0)
    data = pd.merge(data, data[data['d_dt_click'] <= 14].groupby(['role_id', 'item_id'])['cnt'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_14']).fillna(0)
    data = pd.merge(data, data[data['d_dt_click'] <= 21].groupby(['role_id', 'item_id'])['cnt'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_21']).fillna(0)
    data = pd.merge(data, data.groupby(['role_id', 'item_id'])['d_dt_click'].min().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_final'])
    data = pd.merge(data, data.groupby(['role_id', 'item_id'])['d_dt_click'].max().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_first'])

    data['cnt_1_ctr'] = data['cnt_1'] / data[data['d_dt_click'] <= 1].shape[0]
    data['cnt_3_ctr'] = data['cnt_3'] / data[data['d_dt_click'] <= 3].shape[0]
    data['cnt_7_ctr'] = data['cnt_7'] / data[data['d_dt_click'] <= 7].shape[0]
    data['cnt_14_ctr'] = data['cnt_14'] / data[data['d_dt_click'] <= 14].shape[0]
    data['cnt_21_ctr'] = data['cnt_21'] / data[data['d_dt_click'] <= 21].shape[0]

    data.drop_duplicates(['role_id', 'item_id','d_dt_click'], inplace=True)

    data['click_rank_1'] = data['cnt_1'].rank(method='dense', ascending=False)
    data['click_rank_3'] = data['cnt_3'].rank(method='dense', ascending=False)
    data['click_rank_7'] = data['cnt_7'].rank(method='dense', ascending=False)
    data['click_rank_14'] = data['cnt_14'].rank(method='dense', ascending=False)
    data['click_rank_21'] = data['cnt_21'].rank(method='dense', ascending=False)

    data['is_click_dt_1'] = np.where(data['d_dt_click_final'] <= 1,1,0)
    data['is_click_dt_3'] = np.where(data['d_dt_click_final'] <= 3,1,0)
    data['is_click_dt_3'] = np.where(data['d_dt_click_final'] <= 3,1,0)
    data['is_click_dt_7'] = np.where(data['d_dt_click_final'] <= 7,1,0)

    user_feature = pd.merge(user_feature, data[data['d_dt_click'] <= 1].groupby(['role_id'])['cnt'].sum().reset_index(),
                            suffixes=['', '_1'], on='role_id', how='left').fillna(0)
    user_feature = pd.merge(user_feature, data[data['d_dt_click'] <= 3].groupby(['role_id'])['cnt'].sum().reset_index(),
                            suffixes=['', '_3'], on='role_id', how='left').fillna(0)
    user_feature = pd.merge(user_feature, data[data['d_dt_click'] <= 7].groupby(['role_id'])['cnt'].sum().reset_index(),
                            suffixes=['', '_7'], on='role_id', how='left').fillna(0)

    item_feature = pd.merge(item_feature, data[data['d_dt_click'] <= 1].groupby('item_id')['cnt'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'cnt': 'cnt_1_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_click'] <= 3].groupby('item_id')['cnt'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'cnt': 'cnt_3_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_click'] <= 7].groupby('item_id')['cnt'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'cnt': 'cnt_7_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_click'] <= 21].groupby('item_id')['cnt'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'cnt': 'cnt_21_item'})

    user_feature.rename(columns={'cnt_1': 'cnt_1_user', 'cnt_3': 'cnt_3_user', 'cnt_7': 'cnt_7_user'},inplace=True)

    # item_feature.rename(columns={'cnt_1': 'cnt_1_item', 'cnt_3': 'cnt_3_item', 'cnt_7': 'cnt_7_item', 'cnt_14': 'cnt_14_item',
    #              'cnt_21': 'cnt_21_item'},inplace=True)
    # if train_index is not None:
    # data = pd.merge(data, role_data, on=['role_id', 'item_id'], how='inner')
    data.drop_duplicates(['role_id', 'item_id'], keep='last',inplace=True)

    if train_index is not None:
        data = pd.merge(data, train_index.drop(['item_id','label'],axis=1), on=['role_id'], how='inner').fillna(0)
        data = pd.merge(data, train_index, on=['role_id', 'item_id'], how='left').fillna(0)
        data = pd.merge(item_id, data, on='item_id', how='inner')
        data = data.reset_index(drop=True)
        print data['item_id'].nunique()

    #     if sample_rate is not None:
    #         data = pd.concat([data[data['label'] == 1], data[data['label'] == 0]
    #                          .sample(n=sample_rate * train_index.shape[0], random_state=17)], axis=0)
    #     data = data.reset_index(drop=True)
    return data.drop(['d_dt_click','cnt'],axis=1), user_feature, item_feature

def gen_buy_feature(dt, item_id, active_role, user_feature, item_feature):
    data = pd.read_csv('data_0619/src_mall_day_0626.csv', index_col=0)
    # data = pd.read_csv('data_0608/src_mall_day_0615.csv', delimiter='\t', header=None,
    #                    names=['role_id','item_id','buy_num', 'price', 'dt'])
    role_data = data[data['dt'] == (dt - 1)][['role_id', 'item_id']]
    data = data[data['dt']<dt]
    data = pd.merge(active_role, data, on='role_id', how='inner')
    data = pd.merge(item_id, data, on='item_id', how='inner')
    top_pre = list(data[data['dt']>dt-4].groupby('item_id')['buy_num'].sum().reset_index().sort_values(by='buy_num',ascending=False)['item_id'].values[:1])
    global top_pre
    data['d_dt_buy'] = dt - data['dt'].astype(int)
    data = data.drop('dt', axis=1).groupby(['role_id', 'item_id', 'd_dt_buy']).sum().reset_index()
    data = pd.merge(data, data[data['d_dt_buy'] <= 1].groupby(['role_id', 'item_id'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_1']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 3].groupby(['role_id', 'item_id'])['buy_num'].sum().reset_index()
                    , on=['role_id', 'item_id'], how='left', suffixes=['', '_3']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 7].groupby(['role_id', 'item_id'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_7']).fillna(0)
    # data = pd.merge(data, data[data['d_dt_buy'] <= 14].groupby(['role_id', 'item_id'])['buy_num'].sum().reset_index(),
    #                 on=['role_id', 'item_id'], how='left', suffixes=['', '_14']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 21].groupby(['role_id', 'item_id'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_21']).fillna(0)

    data['buy_num_1_ctr'] = data['buy_num_1']/data[data['d_dt_buy']<=1].shape[0]
    data['buy_num_3_ctr'] = data['buy_num_3']/data[data['d_dt_buy']<=3].shape[0]
    data['buy_num_7_ctr'] = data['buy_num_7']/data[data['d_dt_buy']<=7].shape[0]
    # data['buy_num_14_ctr'] = data['buy_num_14']/data[data['d_dt_buy']<=14].shape[0]
    data['buy_num_21_ctr'] = data['buy_num_21']/data[data['d_dt_buy']<=21].shape[0]

    data = pd.merge(data, data.groupby(['role_id', 'item_id'])['d_dt_buy'].min().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_final'])
    # data['buy_rank_1'] = data['buy_num_1'].rank(method='dense', ascending=False)
    # data['buy_rank_3'] = data['buy_num_3'].rank(method='dense', ascending=False)
    # data['buy_rank_7'] = data['buy_num_7'].rank(method='dense', ascending=False)
    # data['buy_rank_14'] = data['buy_num_14'].rank(method='dense', ascending=False)
    # data['buy_rank_21'] = data['buy_num_21'].rank(method='dense', ascending=False)
    data['have_buy_1'] = np.where(data['buy_num_1']>0, 1, 0)
    data['have_buy_3'] = np.where(data['buy_num_3']>0, 1, 0)
    data['have_buy_7'] = np.where(data['buy_num_7']>0, 1, 0)
    data['have_buy_21'] = np.where(data['buy_num_21']>0, 1, 0)
    data.drop_duplicates(['role_id', 'item_id','d_dt_buy'], inplace=True)

    user_feature = pd.merge(user_feature, data[data['d_dt_buy'] <= 1].groupby('role_id')['buy_num'].sum().reset_index(),
                            on='role_id', how='left').fillna(0).rename(columns={'buy_num':'buy_num_1_user'})
    user_feature = pd.merge(user_feature, data[data['d_dt_buy'] <= 3].groupby('role_id')['buy_num'].sum().reset_index(),
                            on='role_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_3_user'})
    user_feature = pd.merge(user_feature, data[data['d_dt_buy'] <= 7].groupby('role_id')['buy_num'].sum().reset_index(),
                            on='role_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_7_user'})
    user_feature = pd.merge(user_feature, data[data['d_dt_buy'] <= 14].groupby('role_id')['buy_num'].sum().reset_index(),
                            on='role_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_14_user'})
    user_feature = pd.merge(user_feature, data[data['d_dt_buy'] <= 21].groupby('role_id')['buy_num'].sum().reset_index(),
                            on='role_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_21_user'})
    #
    # has_buy = data[data['d_dt_buy'] <= 21].groupby(['role_id','item_id'])['buy_num'].sum().unstack().rename(columns=lambda x: str(x)+'_has').reset_index().fillna(0)
    #
    # user_feature = pd.merge(user_feature, has_buy, on='role_id', how='left').fillna(0)

    item_feature = pd.merge(item_feature, data[data['d_dt_buy'] <= 1].groupby('item_id')['buy_num'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_1_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_buy'] <= 3].groupby('item_id')['buy_num'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_3_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_buy'] <= 7].groupby('item_id')['buy_num'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_7_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_buy'] <= 14].groupby('item_id')['buy_num'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_14_item'})
    item_feature = pd.merge(item_feature, data[data['d_dt_buy'] <= 21].groupby('item_id')['buy_num'].sum().reset_index(),
                            on='item_id', how='left').fillna(0).rename(columns={'buy_num': 'buy_num_21_item'})

    # item_feature['buy_num_1_item_rank'] = item_feature['buy_num_1_item'].rank(method='dense', ascending=False)
    # item_feature['buy_num_3_item_rank'] = item_feature['buy_num_3_item'].rank(method='dense', ascending=False)
    # item_feature['buy_num_7_item_rank'] = item_feature['buy_num_7_item'].rank(method='dense', ascending=False)
    # item_feature['buy_num_14_item_rank'] = item_feature['buy_num_14_item'].rank(method='dense', ascending=False)
    # item_feature['buy_num_21_item_rank'] = item_feature['buy_num_21_item'].rank(method='dense', ascending=False)

    # item_feature['item_ctr_7'] = item_feature['cnt_7_item'] / item_feature['buy_num_7_item']
    # item_feature['item_ctr_3'] = item_feature['cnt_3_item'] / item_feature['buy_num_3_item']
    # item_feature['item_ctr_1'] = item_feature['cnt_1_item'] / item_feature['buy_num_1_item']
    # item_feature['item_ctr_21'] = item_feature['cnt_21_item'] / item_feature['buy_num_21_item']
    # item_feature['item_ctr_21'].replace(np.inf, 200)
    # item_feature = item_feature.drop(['buy_num_1_item', 'buy_num_3_item', 'buy_num_7_item', 'buy_num_14_item', 'buy_num_21_item'], axis=1)
    # data = pd.merge(data, role_data, on=['role_id', 'item_id'], how='inner')
    # data.drop_duplicates(['role_id', 'item_id'], inplace=True)
    return data.drop(['d_dt_buy','buy_num'], axis=1), user_feature, item_feature

def gen_user_feature(dt, active_role, enc=None):
    # all date
    filename = 'data_0619/feature_{}.csv'.format(str(dt-1)[-4:])
    data = pd.read_csv(filename, index_col=0)
    data = pd.merge(active_role, data, on='role_id', how='inner')
    #
    # other = pd.read_csv('data_0619/src_appearance_0621_{}_split.csv'.format(str(dt-1)[-4:]), index_col=0)
    # data = pd.merge(data, other.drop(['server','tshirt','pants','hat','hair'],axis=1), on='role_id', how='left')
    # data['gender'].fillna(3, inplace=True)
    # data = pd.concat([data, pd.DataFrame(enc.enc.transform(np.array(data['gender']).reshape(-1, 1)).toarray())],axis=1)
    # data.rename(columns={0:'0_gender',1:'1_gender',2:'2_gender'}, inplace=True)
    return data.drop(['dt','server','os'],axis=1)
    # return data.drop(['dt','server','os'],axis=1)

def gen_item_feature(dt, item_id, enc):
    new_item = pd.read_csv('data_0608/new_item.csv', names=['item_id', 'item_name', 'newdt', 'dt'], delimiter='\t')
    new_item = new_item[new_item['newdt'] > (dt-14)]
    new_item = new_item[new_item['newdt'] < (dt-7)]
    new_item_list = list(new_item['item_id'])
    item_feature = item_id.copy()
    item_feature['is_new'] = item_feature['item_id'].apply(lambda x: x in new_item_list)
    item_feature['is_new'] = np.where(item_feature['is_new'], 1, 0)

    # enc = preprocessing.OneHotEncoder()
    item_feature = pd.concat([item_feature, pd.DataFrame(enc.enc.transform(np.array(item_feature['item_id']).reshape(-1, 1)).toarray())], axis=1)

    return item_feature

def gen_train(dt_start, dt_end):
    global active_role
    train_lgb = pd.DataFrame()
    # #
    for i in range(dt_start, dt_end):
        dt = i
        print 'dt is', dt
        active_role_temp = active_role[active_role['dt'] == dt - 1].drop('dt', axis=1)
        # #
        user_feature = gen_user_feature(dt, active_role_temp)
        item_feature = gen_item_feature(dt, item_id, item_enc)

        train_index = gen_pos_sample_index(dt)
        train_index['label'] = 1

        click_feature, user_feature, item_feature = gen_click_feature(dt, active_role_temp, item_id, user_feature,
                                                                      item_feature, train_index)
        buy_feature, user_feature, item_feature = gen_buy_feature(dt, item_id, active_role_temp, user_feature,
                                                                        item_feature)

        train = pd.merge(click_feature, buy_feature, on=['role_id', 'item_id'], how='outer').fillna(0)

        del click_feature
        del buy_feature
        gc.collect()
        train, user_feature, item_feature = add_fm_vec(train, user_feature, item_feature, dt)

        train = pd.merge(train, user_feature, on='role_id', how='inner')
        train = pd.merge(train, item_feature, on='item_id', how='inner')
        del user_feature
        del item_feature
        gc.collect()
        train_lgb = pd.concat([train_lgb, train])
        del train
    print 'train_shape', train_lgb.shape
    print 'pos_train_shape', train_lgb[train_lgb['label'] == 1].shape

    train_lgb = train_lgb.reset_index(drop=True)
    train_lgb.to_csv('data_0619/train_lgb.csv')
    lgb_model = lgb_train(train_lgb.drop(['role_id', 'item_id'], axis=1))
    save_model('lgb_model', lgb_model)

def gen_test(pre_dt):
    global active_role
    active_role_temp = active_role[active_role['dt'] == pre_dt-1].drop('dt', axis=1)
    # test_role = pd.read_csv('data_0608/src_mall_day_0615.csv', delimiter='\t', header=None,
    #                         names=['role_id', 'item_id', 'buy_num', 'dt'])
    # active_role = test_role[test_role['dt'] == pre_dt]['role_id']
    # active_role = active_role.reset_index().drop('index', axis=1)

    test_user_feature = gen_user_feature(pre_dt, active_role_temp)
    test_item_feature = gen_item_feature(pre_dt, item_id,item_enc)
    test_click_feature, test_user_feature, test_item_feature = gen_click_feature(pre_dt, active_role_temp, item_id,
                                                                                 test_user_feature, test_item_feature)

    test_buy_feature, test_user_feature, test_item_feature = gen_buy_feature(pre_dt, item_id, active_role_temp,
                                                                             test_user_feature, test_item_feature)



    test = pd.merge(test_click_feature, test_buy_feature, on=['role_id', 'item_id'], how='outer').fillna(0)
    del test_click_feature, test_buy_feature
    gc.collect()
    test, test_user_feature, test_item_feature = add_fm_vec(test, test_user_feature, test_item_feature, pre_dt)

    test = pd.merge(test, test_user_feature, on='role_id', how='inner')
    test = pd.merge(test, test_item_feature, on='item_id', how='inner')
    # del test_user_feature, test_item_feature
    gc.collect()

    test_role = pd.read_csv('data_0608/src_mall_day_0615.csv', delimiter='\t', header=None,
                            names=['role_id', 'item_id', 'buy_num', 'dt'])
    test_role = test_role[test_role['dt'] == pre_dt]
    test_role_temp = test_role['role_id'].reset_index().drop('index',axis=1)
    test = pd.merge(test, test_role_temp, on='role_id',how='inner')

    test = test.drop_duplicates(['role_id', 'item_id']).reset_index(drop=True)
    test = pd.merge(test, active_role_temp, how='inner', on='role_id')

    # rec = recommendation_fm()
    # test_rec = rec.train_data('data_0619/buyall_{}.csv'.format(str(pre_dt-1)[-4:]))
    # fm_model = rec.train(test_rec, n_factor=20, epochs=10, lr=0.01, loss='warp')
    # print 'fm_train is over'
    # user_feature_fm = pd.DataFrame(fm_model.user_embeddings)
    # user_feature_fm.rename(columns=lambda x: str(x)+'_user_fm', inplace=True)
    # user_feature_fm = user_feature_fm.reset_index().rename(columns={'index': 'role_id'})
    # user_feature_fm['role_id'] = user_feature_fm['role_id'].apply(lambda x: rec.uid[x])

    # item_feature_fm = pd.DataFrame(fm_model.item_embeddings)
    # item_feature_fm.rename(columns=lambda x: str(x)+'_item_fm', inplace=True)
    # item_feature_fm = item_feature_fm.reset_index().rename(columns={'index': 'item_id'})
    # item_feature_fm['item_id'] = item_feature_fm['item_id'].apply(lambda x: rec.iid[x])

    # test = pd.merge(test, user_feature_fm, on='role_id', how='left', suffixes=['', '_user_fm']).fillna(0)
    # test = pd.merge(test, item_feature_fm, on='item_id', how='left', suffixes=['', '_item_fm']).fillna(0)
    # user_feature_fm = 0
    # item_feature_fm = 0
    # def funcc(a, b):
    #     try:
    #         return fm_model.predict(rec.re_uid[a], [rec.re_iid[b]])[0]
    #     except:
    #         return -2
    # test['fm'] = test.apply(lambda row: funcc(row['role_id'], row['item_id']), axis=1)

    gc.collect()
    # del test_user_feature, test_item_feature
    test.to_csv('data_0619/lgb_test_{}.csv'.format(str(pre_dt)[4:]))
    return test, test_user_feature, test_item_feature

def test_predict(test, test_user_feature, test_item_feature, model, num=1):
    # test_role = pd.read_csv('data_0608/src_mall_day_0615.csv', delimiter='\t', header=None,
    #                         names=['role_id', 'item_id', 'buy_num', 'dt'])
    test_role = pd.read_csv('data_0619/src_mall_day_0626.csv', index_col=0)
    test_role = test_role[test_role['dt'] == pre_dt]
    active_role_temp = active_role[active_role['dt'] == pre_dt-1].drop('dt', axis=1)
    test_role = pd.merge(test_role, active_role_temp, how='inner', on='role_id')
    print list(test_role.columns)

    role_temp = test_role['role_id'].drop_duplicates().reset_index().drop('index', axis=1)
    item_temp = item_id.copy()
    role_temp['key'] = 0
    item_temp['key'] = 0
    test_temp = pd.merge(role_temp, item_temp, on='key').drop('key', axis=1)
    test_temp = pd.merge(test_temp, test_user_feature, on='role_id', how='left')
    test_temp = pd.merge(test_temp, test_item_feature, on='item_id', how='left')

    test_temp = pd.merge(test_temp, test, on=list(test_temp.columns), how='left').fillna(0)
    test_temp = test_temp[list(test.columns)]
    # test_temp.to_csv('data_0619/lgb_test_temp_{}.csv'.format(str(pre_dt)[4:]))

    if num == 1:
        test['preb'] = model.predict(test.drop(['role_id','item_id'],axis=1))
    else:
        test_temp['preb'] = model.predict(test_temp.drop(['role_id', 'item_id'], axis=1))

    test_role_temp = test_role[test_role['dt'] == pre_dt]
    new_item = 0

    buy_item = pd.read_csv('data_0619/buyall_{}.csv'.format(str(pre_dt-1)[-4:]),index_col=0)
    buy_item = pd.merge(buy_item, item_id, on='item_id', how='inner')

    info = predict_info()
    for role in test_role_temp['role_id'].unique():
        role_buy_item = list(buy_item[buy_item['role_id']==role]['item_id'].values)
        if num == 1:
            # pre = list(test[test['role_id'] == role].sort_values(by=['preb'], ascending=False).reset_index(drop=True)[
            #                'item_id'])
            pre = rec_pre(test[test['role_id'] == role][['item_id', 'preb']], item_id, 0.05)
        if num != 1:
            pre = list(test_temp[test_temp['role_id'] == role].sort_values(by=['preb'], ascending=False).reset_index(
                drop=True)['item_id'])
            pre = [x for x in pre if x not in top_pre and x not in role_buy_item]
            pre = top_pre+pre
        # print pre

        for item,price in test_role[test_role['role_id'] == role][['item_id','price']].values:
            if item in [978,988,989,990,991, 992, 993, 994, 995]:
                new_item += 1
                continue
            try:
                print item, pre.index(item)
                info.add(pre.index(item),price)
            except:
                print 'wrong'

    info.print_result()
    print 'new_item_number', new_item
    gc.collect()
    return test

if __name__ == '__main__':
    import gc
    active_role = pd.read_csv('data_0619/active_role_0627.csv', delimiter='\t', names=['role_id', 'dt'])
    item_id = pd.read_csv('data_0608/item_id.csv', names=['item_id'])
    item_id = item_id[item_id['item_id'].apply(lambda x: x not in [978,988,989,990,991, 992, 993, 994, 995])]
    item_id = item_id.reset_index(drop=True)
    # gender_enc = onehotcode(np.array([1, 2, 3]).reshape(-1, 1))
    item_enc = onehotcode(np.array(item_id['item_id']).reshape(-1, 1))
    gen_train(20180515, 20180516)
    lgb_model = load_model('lgb_model')
    # test part
    pre_dt = 20180516
    test, test_user_feature, test_item_feature= gen_test(pre_dt)
    print 'test_shape',test.shape
    test_predict(test, test_user_feature, test_item_feature, lgb_model,2)
