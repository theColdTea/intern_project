


def item_feature_construct(rec, fm_model):
    item_feature_fm = pd.DataFrame(fm_model.item_embeddings).reset_index().rename(columns={'index': 'item_id'})
    item_feature_fm['item_id'] = item_feature_fm['item_id'].apply(lambda x: rec.iid[x])

    return item_feature_fm

def user_feature_construct(rec, fm_model):
    data_feature_name = pd.read_csv('data_0608/feature_0525_col.csv', delimiter='\t', names=[''])
    data = pd.read_csv('data_0608/feature_0525.csv', delimiter='\t',
                            names=[x.strip() for x in list(data_feature_name.reset_index()['level_0'])[:-5]])
    #
    rec = recommendation()
    train_data = rec.train_data('data_0608/test_buy_25.csv')
    fm_model = rec.train(train_data, n_factor=10, epochs=20, lr=0.001, loss='warp')
    print 'fm_train is over'
    #
    user_feature_fm = pd.DataFrame(fm_model.user_embeddings).reset_index().rename(columns={'index': 'role_id'})
    user_feature_fm['role_id'] = user_feature_fm['role_id'].apply(lambda x: rec.uid[x])
    # fillna(0)???
    data = pd.merge(data, user_feature_fm, on='role_id', how='left').fillna(0)
    return data.drop(['dt','os','server'],axis=1)


def click_feature(active_role, item_id):
    data = pd.read_csv('data_0608/data_click_role_item_0525.csv',index_col=0)
    data = pd.merge(active_role, data, on='role_id', how='inner')
    data = pd.merge(item_id, data, on='item_id', how='inner')
    data['d_dt'] = 20180526-data['dt'].astype(int)
    data = data.drop('dt',axis=1).groupby(['role_id','item_id','d_dt']).sum().reset_index()
    data = pd.merge(data, data[data['d_dt']<=1].groupby(['role_id','item_id','d_dt'])['cnt'].sum().reset_index(),
                    on=['role_id','item_id','d_dt'], how='left',suffixes=['','_1']).fillna(0)
    data = pd.merge(data, data[data['d_dt']<=3].groupby(['role_id','item_id','d_dt'])['cnt'].sum().reset_index()
                    , on=['role_id','item_id','d_dt'], how='left',suffixes=['','_3']).fillna(0)
    data = pd.merge(data, data[data['d_dt']<=7].groupby(['role_id','item_id','d_dt'])['cnt'].sum().reset_index(),
                    on=['role_id','item_id','d_dt'], how='left',suffixes=['','_week']).fillna(0)
    data = pd.merge(data, data[data['d_dt']<=30].groupby(['role_id','item_id','d_dt'])['cnt'].sum().reset_index(),
                    on=['role_id','item_id','d_dt'], how='left',suffixes=['','_month']).fillna(0)
    data = pd.merge(data, data.groupby(['role_id','item_id'])['d_dt'].min().reset_index(),
                                      on=['role_id','item_id'], how='left', suffixes=['','_final_click']).fillna(100)
    data['final_click_dt_1'] = np.where(data['d_dt_final_click'] <= 1,1,0)
    data['final_click_dt_3'] = np.where(data['d_dt_final_click'] <= 3,1,0)
    data['final_click_dt_week'] = np.where(data['d_dt_final_click'] <= 7,1,0)
    data['rank_week'] = data['cnt_week'].rank(method='min',ascending=False)

    return data.drop('d_dt',axis=1)

def buy_feature(active_role, item_id):
    data = pd.read_csv('data_0608/src_mall_day_0615.csv',delimiter='\t',header=None,names=['role_id','item_id','buy_num','dt'])
    data = pd.merge(active_role, data, on='role_id', how='inner')
    data = pd.merge(item_id, data, on='item_id', how='inner')

    data['d_dt_buy'] = 20180526-data['dt'].astype(int)
    data = data.drop('dt', axis=1).groupby(['role_id', 'item_id', 'd_dt_buy']).sum().reset_index()
    data = pd.merge(data, data[data['d_dt_buy'] <= 1].groupby(['role_id', 'item_id', 'd_dt_buy'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id', 'd_dt_buy'], how='left', suffixes=['', '_1']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 3].groupby(['role_id', 'item_id', 'd_dt_buy'])['buy_num'].sum().reset_index()
                    , on=['role_id', 'item_id', 'd_dt_buy'], how='left', suffixes=['', '_3']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 7].groupby(['role_id', 'item_id', 'd_dt_buy'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id', 'd_dt_buy'], how='left', suffixes=['', '_week']).fillna(0)
    data = pd.merge(data, data[data['d_dt_buy'] <= 30].groupby(['role_id', 'item_id', 'd_dt_buy'])['buy_num'].sum().reset_index(),
                    on=['role_id', 'item_id', 'd_dt_buy'], how='left', suffixes=['', '_month']).fillna(0)
    data = pd.merge(data, data.groupby(['role_id', 'item_id'])['d_dt_buy'].min().reset_index(),
                    on=['role_id', 'item_id'], how='left', suffixes=['', '_final']).fillna(100)
    data['final_buy_dt_1'] = np.where(data['d_dt_buy_final'] <= 1, 1, 0)
    data['final_buy_dt_3'] = np.where(data['d_dt_buy_final'] <= 3,1,0)
    data['final_buy_dt_3'] = np.where(data['d_dt_buy_final'] <= 3,1,0)

    data['buy_rank_week'] = data['buy_num_week'].rank(method='min', ascending=False)

    return data.drop('d_dt_buy',axis=1)


def cross_feature_construct(active_role):
    item_id = pd.read_csv('data_0608/item_id.csv', names=['item_id'])
    click_data = click_feature(active_role,item_id)
    buy_data = buy_feature(active_role,item_id)
    del(item_id)
    # filter the active role
    data = pd.merge(click_data, buy_data, on=['role_id','item_id'], how='left')
    data['ctr'] = data['d_dt_buy_final']/data['d_dt_final_click']

    data['label'] = np.where(data['d_dt_buy_final'].isnull(),0,1)
    data.fillna(0,inplace=True)

    return data


cor_1,cor_3,cor_5,cor_10,cor_15,cor_20,cor_25,tol = [0,0,0,0,0,0,0,0]
pri_1,pri_3,pri_5,pri_10,pri_15,pri_20,pri_25 = [0,0,0,0,0,0,0]
def init_cor():
    global cor_1, cor_3, cor_5, cor_10, cor_15, cor_20, cor_25, tol
    global pri_1,pri_3,pri_5,pri_10,pri_15,pri_20,pri_25
    cor_1 = 0
    cor_3 = 0
    cor_5 = 0
    cor_10 = 0
    cor_15 = 0
    cor_20 = 0
    cor_25 = 0
    tol = 0
    pri_1, pri_3, pri_5, pri_10, pri_15, pri_20, pri_25 = [0, 0, 0, 0, 0, 0, 0]

def add_cor(k,price):
    global cor_1, cor_3, cor_5, cor_10, cor_15, cor_20, cor_25, tol
    global pri_1,pri_3,pri_5,pri_10,pri_15,pri_20,pri_25
    if k < 25:
        cor_25 += 1
        pri_25+=price
    if k < 20:
        cor_20 += 1
        pri_20 += price
    if k < 15:
        cor_15 += 1
        pri_15 += price
    if k < 10:
        cor_10 += 1
        pri_10 += price
    if k < 5:
        cor_5 += 1
        pri_5 += price
    if k < 3:
        cor_3 += 1
        pri_3 += price
    if k < 1:
        cor_1 += 1
        pri_1 += price
    tol+=1
