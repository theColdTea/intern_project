#coding:utf8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import datetime
import json
#import pandas as pd
#from pyspark.ml.linalg import DenseVector, Vectors
#cols_consume=["new","mallCloth","mallGift","advSupply","classicSupply","gunSupply"]

##########################
# role - item  feature
#
##########################



def get_feature_role_mallItem(spark,vDate,day_num=7,min_open_value=5):
    '''
        1. 点击次数：  
        2. 购买次数： 
        3. 点击次数排名 
        4. 购买的天数、点击的天数、cvr：购买的天数/点击的天数 
        5. 最后一天在线是否点击，是否购买 
    '''
    vDateB = (datetime.datetime.strptime(vDate, "%Y%m%d")-datetime.timedelta(days=(day_num-1))).strftime('%Y%m%d')
    spark.sql("add file /home/scheduler.run/us/jobs/h45/python/click_dict_to_item.py")
    sql_str='''
        select
            role_item_click_week.role_id as role_id,role_item_click_week.item_id as item_id,
            --buy
            coalesce(buy_num_day,-1) as buy_num_day,coalesce(buy_num_day3,-1) as buy_num_day3,coalesce(buy_num_week,-1) as buy_num_week,coalesce(buy_num_month,-1) as buy_num_month,
            coalesce(buy_day_week,-1) as buy_day_week, coalesce(buy_day_month,-1) as buy_day_month,
            --click
            coalesce(final_click_day,100),
            coalesce(click_num_day,0) as click_num_day,coalesce(click_num_day3,0) as click_num_day3,coalesce(click_num_week,0) as click_num_week,coalesce(click_num_month,0) as click_num_month,
            coalesce(click_day_week,0) as click_day_week, coalesce(click_day_month,0) as click_day_month,
            coalesce(click_num_dis_day,0) as click_num_dis_day,
            coalesce(click_num_dis_day3,0) as click_num_dis_day3,
            coalesce(click_num_dis_week,0) as click_num_dis_week,
            coalesce(click_num_dis_month,0) as click_num_dis_month,

            coalesce(rn_click_week,-1) as rn_click_week,
            coalesce(rn_buy_week,-1) as rn_buy_week,
            coalesce(final_buy_day,100) as final_buy_day,
            coalesce(role_final_day-final_buy_day,-1) final_buy_gap,
            role_final_day,
            --open
            coalesce(UIMallCloth_day,0) UIMallCloth_day,
            coalesce(UIMallCloth_day3,0) UIMallCloth_day3,
            coalesce(UIMallCloth_week,0) UIMallCloth_week,
            coalesce(UIMallCloth_month,0) UIMallCloth_month,
            coalesce(UIMall_day,0) UIMall_day,
            coalesce(UIMall_day3,0) UIMall_day3,
            coalesce(UIMall_week,0) UIMall_week,
            coalesce(UIMall_month,0) UIMall_month,

            if(UIMallCloth_day is null or UIMallCloth_day<5,0,coalesce(click_num_dis_day,0)/UIMallCloth_day) as UIMallCloth_ctr_day,
            if(UIMallCloth_day3 is null or UIMallCloth_day3<5,0,coalesce(click_num_dis_day3,0)/UIMallCloth_day3) as UIMallCloth_ctr_day3,
            if(UIMallCloth_week is null or UIMallCloth_week<5,0,coalesce(click_num_dis_week,0)/UIMallCloth_week) as UIMallCloth_ctr_week,
            if(UIMallCloth_month is null or UIMallCloth_month<5,0,coalesce(click_num_dis_month,0)/UIMallCloth_month) as UIMallCloth_ctr_month,

            if(UIMall_day is null or UIMall_day<3,0,coalesce(click_num_dis_day,0)/UIMall_day) as UIMall_ctr_day,
            if(UIMall_day3 is null or UIMall_day3<4,0,coalesce(click_num_dis_day3,0)/UIMall_day3) as UIMall_ctr_day3,
            if(UIMall_week is null or UIMall_week<6,0,coalesce(click_num_dis_week,0)/UIMall_week) as UIMall_ctr_week,
            if(UIMall_month is null or UIMall_month<10,0,coalesce(click_num_dis_month,0)/UIMall_month) as UIMall_ctr_month
        from
        (
            select
                role_id,item_id,sum(if(dt<=1,click_num_day,0)) as click_num_day,sum(if(dt<=3,click_num_day,0)) as click_num_day3,
                sum(if(dt<=7,click_num_day,0)) as click_num_week,sum(click_num_day) as click_num_month,

                sum(if(dt<=1,click_num_dis_day,0)) as click_num_dis_day,sum(if(dt<=3,click_num_dis_day,0)) as click_num_dis_day3,
                sum(if(dt<=7,click_num_dis_day,0)) as click_num_dis_week,sum(click_num_dis_day) as click_num_dis_month,

                count(distinct dt) as click_day_month,count(if(dt<=7,dt,100))-1 as click_day_week,
                min(dt) as final_click_day,
                row_number() over (distribute by role_id sort by sum(if(dt<=7,click_num_day,0)) DESC)  as rn_click_week
            from
            (
                select
                    datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(dt,'yyyyMMdd'),'yyyy-MM-dd')) as dt,
                    role_id,item_id,sum(int(click_num)) as click_num_day,sum(if(child_tab!='UIDtsAppearanceMall',1,0)) as click_num_dis_day
                from
                (
                    select
                        transform (item_list,role_id,child_tab,dt) using 'python click_dict_to_item.py' as (item_id,click_num,role_id,child_tab,dt)
                    from
                        us_h45na_default.src_mall_click_day
                    where
                        dt<=${vDate} and dt>=${vDateB}  and item_list!='{}' and ( child_tab='UIMallCloth'   or child_tab='UIMall' or child_tab='UIDtsAppearanceMall')
                ) a
                group by 
                    role_id,item_id,datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(dt,'yyyyMMdd'),'yyyy-MM-dd'))
            ) a
            group by role_id,item_id
        ) role_item_click_week
        left join
        (
            select
                role_id,item_id,sum(if(dt<=1,buy_num_day,0)) as buy_num_day,sum(if(dt<=3,buy_num_day,0)) as buy_num_day3,sum(if(dt<=7,buy_num_day,0))  as buy_num_week,sum(buy_num_day) as buy_num_month,
                count(distinct dt) as buy_day_month,count(if(dt<=7,dt,100))-1 as buy_day_week,
                min(dt) as final_buy_day,
                row_number() over (distribute by role_id sort by sum(if(dt<=7,buy_num_day,0)) DESC) as rn_buy_week
            from
            (
                select
                    datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(dt,'yyyyMMdd'),'yyyy-MM-dd')) as dt,
                    role_id,item_id,count(1) as buy_num_day
                from
                    us_h45na_default.src_mall_day
                where
                    dt<=${vDate} and dt>=${vDateB} and ( mall_type='MallCloth'  or mall_type='MallGiftBag' or  mall_type='AppearanceMallHot')
                group by role_id,item_id,dt
            ) a
            group by role_id,item_id
        ) role_item_buy_week
        on 
            role_item_buy_week.role_id=role_item_click_week.role_id and role_item_buy_week.item_id=role_item_click_week.item_id
        left join
        (
            select
                role_id,datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(role_final_dt,'yyyyMMdd'),'yyyy-MM-dd')) as role_final_day
            from
                us_h45na_default.ods_p1_all
            where
                dt=${vDate}
        ) online
        on
            role_item_click_week.role_id=online.role_id
        left join
        (
            select
                role_id,
                sum(if(child_tab="UIMallCloth" and dt<1,open_num_day,0)) UIMallCloth_day,
                sum(if(child_tab="UIMallCloth" and dt<3,open_num_day,0)) UIMallCloth_day3,
                sum(if(child_tab="UIMallCloth" and dt<7,open_num_day,0)) UIMallCloth_week,
                sum(if(child_tab="UIMallCloth",open_num_day,0)) UIMallCloth_month,

                sum(if(child_tab="UIMall" and dt<1,open_num_day,0)) UIMall_day,
                sum(if(child_tab="UIMall" and dt<3,open_num_day,0)) UIMall_day3,
                sum(if(child_tab="UIMall" and dt<7,open_num_day,0)) UIMall_week,
                sum(if(child_tab="UIMall",open_num_day,0)) UIMall_month
            from
            (
                select
                    role_id,child_tab,
                    datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(dt,'yyyyMMdd'),'yyyy-MM-dd')) as dt,
                    count(1) as open_num_day
                from
                    us_h45na_default.src_mall_click_day
                where
                    dt<=${vDate} and dt>=${vDateB}  and item_list!='{}' and ( child_tab='UIMallCloth'   or child_tab='UIMall')
                group by 
                    role_id,child_tab,datediff(from_unixtime(unix_timestamp('${vDate}','yyyyMMdd'),'yyyy-MM-dd'),from_unixtime(unix_timestamp(dt,'yyyyMMdd'),'yyyy-MM-dd'))
            ) a
            group by 
                role_id
        ) role_open
        on
            role_item_click_week.role_id=role_open.role_id
    '''
    sql_str=sql_str.replace("${vDate}",vDate)
    sql_str=sql_str.replace("${vDateB}",vDateB)
    print sql_str
    df=spark.sql(sql_str)
    return df


