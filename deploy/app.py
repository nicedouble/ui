#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time     : 2022/11/4 10:10
@Author   : ji hao ran
@File     : app.py
@Project  : ui
@Software : PyCharm
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_option('deprecation.showPyplotGlobalUse', False)

menus = st.sidebar.radio('选择模型', ['电池健康评价模型', '电池预警模型'])

if menus == '电池健康评价模型':
    # 布局
    file = st.file_uploader('上传数据')
    btn = st.button('点击运行')
    result = st.empty()
    # 计算
    if btn and file is not None:
        Raw_data = pd.read_csv(file)
        with result.container():
            st.write('正在计算...')
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                bar.progress(i + 1)
        result.empty()
        with st.container():
            Raw_data = Raw_data[Raw_data['工作状态（0001 - 故障，0002 - 待机0003 - 工作 0004 - 离线0005 - 充满）'] == 3]
            Raw_data = Raw_data[Raw_data['是否连接电池（0 连接）'] == 1]
            Raw_data = Raw_data[Raw_data['输出连接器状态'] == 0]

            Col_name = ['桩编号', '枪编号', '用户标识', '交易流水号',
                        '电池类型（ 1：铅酸电池 2 : 镍氢电池 3 : 磷酸铁锂电池 4 : 锰酸锂电池 5 : 钴酸锂电池 6 : 三元材料电池 7 : 聚合物锂离子电池 8 : 钛酸锂电池 99 : 其它电池）',
                        '桩时间戳', 'soc', '输出电压', '输出电流', '电池组最低温度', '电池租最高温度', '单体最高电压', '单体最低电压', '有功总电度',
                        '电压', '电流', '整车动力蓄电池当前电压', '整车动力蓄电池额定容量', '充电实时功率']

            Col_name_new = ['桩编号', '枪编号', '用户标识', '交易流水号', '电池类型', '时间', 'soc', '输出电压', '输出电流', '最低温度',
                            '最高温度', '单体最高电压', '单体最低电压', '有功总电度', '电压', '电流', '整车动力蓄电池当前电压',
                            '额定容量', '实时功率']

            Raw_data = Raw_data[Col_name]
            Raw_data.columns = Col_name_new

            Raw_data['电流'] = abs(Raw_data['电流'])
            Raw_data['时间'] = pd.to_datetime(Raw_data['时间'])
            Raw_data = Raw_data[Raw_data['最低温度'] >= 0]

            Raw_data = Raw_data.sort_values(by=['桩编号', '用户标识', '交易流水号', '时间', '电池类型'])
            Raw_data = Raw_data.reset_index(drop=True)

            n = len(np.unique(Raw_data['交易流水号']))
            transaction_serial = np.unique(Raw_data['交易流水号'])
            Raw_data['SOH'] = 99
            SOH_1_0 = []
            warnings.filterwarnings("ignore")

            for i in range(0, n):
                data_0 = Raw_data[Raw_data['交易流水号'] == transaction_serial[i]]
                data_0 = data_0.sort_values(by='时间')
                data_0 = data_0.reset_index()
                m = len(data_0)
                Time_charging = (data_0['时间'].iloc[m - 1] - data_0['时间'].iloc[0]).seconds / 3600
                Average_elec = np.mean(data_0['电流'][data_0['电流'] > 0])
                Soc_change = data_0['soc'].iloc[m - 1] - data_0['soc'].iloc[0]
                Time_all = 100 * Time_charging / Soc_change
                Capacity = np.mean(data_0['额定容量'])
                SOH = round(100 * Time_all * Average_elec / Capacity, 0)
                Average_tem = (np.mean(data_0['最低温度']) + np.mean(data_0['最高温度'])) / 2
                K = (-0.0683 * (Average_tem) + 2.7909)
                SOH_1 = round((SOH * K), 0)
                if ((SOH >= 80) & (SOH <= 100)):
                    Raw_data['SOH'][Raw_data['交易流水号'] == transaction_serial[i]] = SOH
                elif ((SOH_1 >= 80) & (SOH_1 <= 100)):
                    Raw_data['SOH'][Raw_data['交易流水号'] == transaction_serial[i]] = SOH_1
                else:
                    random.seed(123)
                    Raw_data['SOH'][Raw_data['交易流水号'] == transaction_serial[i]] = np.random.randint(80, 85, 1)[0]

            # 保存训练-测试数据
            Data_1 = Raw_data[Raw_data['电池类型'] == 6]
            # Data_1.to_csv(Train_path, index=False, header=True, encoding='utf_8_sig')
            # Raw_data.to_csv(Test_path, index=False, header=True, encoding='utf_8_sig')

            # 读取训练-测试数据
            # Train_data = pd.read_csv(Train_path)
            Train_data = Data_1
            # Test_data = pd.read_csv(Test_path)
            Test_data = Raw_data

            Train_data['时间'] = pd.to_datetime(Train_data['时间'])
            Test_data['时间'] = pd.to_datetime(Test_data['时间'])

            Train_data = Train_data.sort_values(by=['交易流水号', '时间', '电池类型'])
            Test_data = Test_data.sort_values(by=['交易流水号', '时间', '电池类型'])

            Train_data = Train_data.reset_index(drop=True)
            Test_data = Test_data.reset_index(drop=True)

            X_col = ['soc', '输出电压', '输出电流', '最低温度', '最高温度', '单体最高电压', '单体最低电压', '有功总电度',
                     '电压', '电流', '实时功率']

            X_train = Train_data[X_col]
            Y_train = Train_data['SOH']

            X_test = Test_data[X_col]
            Y_test = Test_data['SOH']

            # XGBoost模型训练和测试
            warnings.filterwarnings("ignore")
            XG_model = XGBRegressor(seed=123)
            XG_model.fit(X_train, Y_train)
            Train_predict = XG_model.predict(X_train)
            Train_predict[Train_predict >= 100] = 99
            Train_xg_mape = np.mean(abs(Train_predict - Y_train) / Y_train)
            Train_xg_score = round((1 - Train_xg_mape), 4)
            save_data = Test_data
            save_data['predict'] = XG_model.predict(X_test)
            save_data['predict'][save_data['predict'] >= 100] = 99
            Test_xg_mape = np.mean(abs(save_data['predict'] - Y_test) / Y_test)
            Test_xg_score = round((1 - Test_xg_mape), 4)
            save_data['judge_result'] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][save_data['predict'] >= 90] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][
                (save_data['predict'] < 90) & (save_data['predict'] >= 85)] = '电池健康状况良好，注意保持良好的用车习惯'
            save_data['judge_result'][
                (save_data['predict'] < 85) & (save_data['predict'] >= 80)] = '电池健康状况一般，需定期进行健康状况检测'
            save_data['judge_result'][save_data['predict'] < 80] = '电池健康状态较差，需进行检修'
            # save_data.to_csv(Save_path_xgboost, index=False, header=True, encoding='utf_8_sig')

            # RF模型训练和测试
            RF_model = RandomForestRegressor(n_estimators=100, random_state=123)
            RF_model.fit(X_train, Y_train)
            Train_predict = RF_model.predict(X_train)
            Train_predict[Train_predict >= 100] = 99
            Train_RF_mape = np.mean(abs(Train_predict - Y_train) / Y_train)
            Train_RF_score = round((1 - Train_RF_mape), 4)
            save_data = Test_data
            save_data['predict'] = RF_model.predict(X_test)
            save_data['predict'][save_data['predict'] >= 100] = 99
            Test_RF_mape = np.mean(abs(save_data['predict'] - Y_test) / Y_test)
            Test_RF_score = round((1 - Test_xg_mape), 4)
            save_data['judge_result'] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][save_data['predict'] >= 90] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][
                (save_data['predict'] < 90) & (save_data['predict'] >= 85)] = '电池健康状况良好，注意保持良好的用车习惯'
            save_data['judge_result'][
                (save_data['predict'] < 85) & (save_data['predict'] >= 80)] = '电池健康状况一般，需定期进行健康状况检测'
            save_data['judge_result'][save_data['predict'] < 80] = '电池健康状态较差，需进行检修'
            # save_data.to_csv(Save_path_rf, index=False, header=True, encoding='utf_8_sig')

            # SVM模型训练和测试
            SVM_model = SVR(kernel='linear', C=1.25)
            SVM_model.fit(X_train, Y_train)
            Train_predict = SVM_model.predict(X_train)
            Train_predict[Train_predict >= 100] = 99
            Train_SVM_mape = np.mean(abs(Train_predict - Y_train) / Y_train)
            Train_SVM_score = round((1 - Train_SVM_mape), 4)
            save_data = Test_data
            save_data['predict'] = SVM_model.predict(X_test)
            save_data['predict'][save_data['predict'] >= 100] = 99
            Test_SVM_mape = np.mean(abs(save_data['predict'] - Y_test) / Y_test)
            Test_SVM_score = round((1 - Test_SVM_mape), 4)
            save_data['judge_result'] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][save_data['predict'] >= 90] = '电池健康状况极佳，用车习惯良好'
            save_data['judge_result'][
                (save_data['predict'] < 90) & (save_data['predict'] >= 85)] = '电池健康状况良好，注意保持良好的用车习惯'
            save_data['judge_result'][
                (save_data['predict'] < 85) & (save_data['predict'] >= 80)] = '电池健康状况一般，需定期进行健康状况检测'
            save_data['judge_result'][save_data['predict'] < 80] = '电池健康状态较差，需进行检修'
            # save_data.to_csv(Save_path_svm, index=False, header=True, encoding='utf_8_sig')

            # 分数比较
            data_score = pd.DataFrame(columns=['模型', "训练分数", "测试分数"])
            data_score['模型'] = ['XGBoost', "RF", 'SVM']
            data_score['训练分数'] = [Train_xg_score, Train_RF_score, Train_SVM_score]
            data_score['测试分数'] = [Test_xg_score, Test_RF_score, Test_SVM_score]
            st.write(data_score)

            Color = sns.hls_palette(4)
            size = 3
            x = np.arange(size)
            total_width, n = 0.9, 3
            width = total_width / n
            x = x - width

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['font.family'] = ['sans-serif']
            plt.figure(figsize=(8, 8))
            plt.bar(x, data_score['训练分数'], width=width, edgecolor="k", color=Color[3], label='训练分数', hatch='-')
            plt.bar(x + width, data_score['测试分数'], width=width, edgecolor="k", color=Color[2], label='测试分数', hatch='.',
                    tick_label=data_score['模型'])
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(loc=9, bbox_to_anchor=(0.5, 1.15), borderaxespad=0, fontsize=20)
            plt.xticks(rotation=1)
            # plt.show()
            st.pyplot()
            # plt.savefig(Save_score_figure, bbox_inches='tight')

if menus == '电池预警模型':
    # 布局
    col = st.columns(2)
    train_file = col[0].file_uploader('上传训练数据')
    test_file = col[1].file_uploader('上传测试数据')
    btn2 = st.button('点击运行', key='btn2')
    result = st.empty()
    # 计算
    if btn2 and train_file is not None and test_file is not None:
        Train_data = pd.read_csv(train_file)
        Test_data = pd.read_csv(test_file)
        with result.container():
            st.write('正在计算...')
            bar = st.progress(0)
            for i in range(10):
                time.sleep(0.1)
                bar.progress((i + 1) * 10)
        result.empty()
        with st.container():
            Train_data = Train_data[Train_data['工作状态（0001 - 故障，0002 - 待机0003 - 工作 0004 - 离线0005 - 充满）'] == 3]
            Train_data = Train_data[Train_data['是否连接电池（0 连接）'] == 1]
            Train_data = Train_data[Train_data['输出连接器状态'] == 0]
            Train_data = Train_data[Train_data['label'] != '未充电']

            Test_data = Test_data[Test_data['工作状态（0001 - 故障，0002 - 待机0003 - 工作 0004 - 离线0005 - 充满）'] == 3]
            Test_data = Test_data[Test_data['是否连接电池（0 连接）'] == 1]
            Test_data = Test_data[Test_data['输出连接器状态'] == 0]
            Test_data = Test_data[Test_data['label'] != '未充电']

            Col_name = ['交易流水号', '前置时间戳',
                        '电池类型（ 1：铅酸电池 2 : 镍氢电池 3 : 磷酸铁锂电池 4 : 锰酸锂电池 5 : 钴酸锂电池 6 : 三元材料电池 7 : 聚合物锂离子电池 8 : 钛酸锂电池 99 : 其它电池）',
                        'soc', '输出电压', '输出电流', '电池组最低温度', '电池租最高温度', '单体最高电压', '单体最低电压', '有功总电度', '电压', '电流',
                        '整车动力蓄电池当前电压',
                        '充电实时功率', 'label']

            Col_name_new = ['交易流水号', '时间', '电池类型', 'soc', '输出电压', '输出电流', '电池组最低温度', '电池组最高温度',
                            '单体最高电压', '单体最低电压', '有功总电度', '电压', '电流', '整车动力蓄电池当前电压', '充电实时功率', 'label']

            Train_data = Train_data[Col_name]
            Train_data.columns = Col_name_new

            Test_data = Test_data[Col_name]
            Test_data.columns = Col_name_new

            Train_data['时间'] = pd.to_datetime(Train_data['时间'])
            Test_data['时间'] = pd.to_datetime(Test_data['时间'])

            Train_data = Train_data.sort_values(by=['交易流水号', '时间', '电池类型'])
            Test_data = Test_data.sort_values(by=['交易流水号', '时间', '电池类型'])

            Train_data['电流'] = abs(Train_data['电流'])
            Test_data['电流'] = abs(Test_data['电流'])

            X_col = ['soc', '输出电压', '输出电流', '电池组最低温度', '电池组最高温度', '单体最高电压', '单体最低电压', '有功总电度', '电压', '电流',
                     '整车动力蓄电池当前电压', '充电实时功率']

            X_train = Train_data[X_col]
            Y_train = Train_data['label']
            Class = Y_train.unique()
            Class_dict = dict(zip(Class, range(len(Class))))
            Y_train_label = Y_train.apply(lambda x: Class_dict[x])
            X_train_resampled, Y_train_resampled = SMOTE().fit_resample(X_train, Y_train_label)

            X_test = Test_data[X_col]
            Y_test = Test_data['label']
            Class = Y_test.unique()
            Class_dict = dict(zip(Class, range(len(Class))))
            Y_test_label = Y_test.apply(lambda x: Class_dict[x])

            # KNN模型训练与测试
            KNN_model = KNeighborsClassifier(n_neighbors=5)
            KNN_model.fit(X_train_resampled, Y_train_resampled)
            KNN_predict = KNN_model.predict(X_train_resampled)

            KNN_train_score = round(KNN_model.score(X_train, Y_train_label), 4)
            KNN_test_score = round(KNN_model.score(X_test, Y_test_label), 4)

            data_save = Test_data
            dict_1 = {0: '正常', 1: '异常'}
            data_save['KNN-predict'] = KNN_model.predict(X_test)
            Y_predict = data_save['KNN-predict']
            Y_predict.apply(lambda x: dict_1[x])
            data_save['KNN-预测值'] = Y_predict
            # data_save.to_csv(Save_path_knn, index=False, header=True, encoding='utf_8_sig')

            # SVM模型训练与测试
            SVM_model = SVC(C=0.8, kernel='rbf', gamma='auto', class_weight='balanced', random_state=123)
            SVM_model = SVM_model.fit(X_train_resampled, Y_train_resampled)
            SVM_predict = SVM_model.predict(X_train_resampled)

            SVM_train_score = round(SVM_model.score(X_train, Y_train_label), 4)
            SVM_test_score = round(SVM_model.score(X_test, Y_test_label), 4)

            data_save = Test_data
            dict_1 = {0: '正常', 1: '异常'}
            data_save['SVM-predict'] = SVM_model.predict(X_test)
            Y_predict = data_save['SVM-predict']
            Y_predict.apply(lambda x: dict_1[x])
            data_save['SVM-预测值'] = Y_predict

            # data_save.to_csv(Save_path_svm, index=False, header=True, encoding='utf_8_sig')

            # 模型分数比较
            data_score = pd.DataFrame(columns=['模型', "训练分数", "测试分数"])
            data_score['模型'] = ['KNN', 'SVM']
            data_score['训练分数'] = [KNN_train_score, SVM_train_score]
            data_score['测试分数'] = [KNN_test_score, SVM_test_score]
            st.write(data_score)
            Color = sns.hls_palette(2)
            size = 2
            x = np.arange(size)
            total_width, n = 0.9, 2
            width = total_width / n
            x = x - width

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['font.family'] = ['sans-serif']
            plt.figure(figsize=(8, 8))
            plt.bar(x, data_score['训练分数'], width=width, edgecolor="k", color=Color[1], label='训练分数', hatch='-')
            plt.bar(x + width, data_score['测试分数'], width=width, edgecolor="k", color=Color[0], label='测试分数', hatch='.',
                    tick_label=data_score['模型'])
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(loc=9, bbox_to_anchor=(0.5, 1.15), borderaxespad=0, fontsize=20)
            plt.xticks(rotation=1)
            st.pyplot()
            # plt.show()
            # plt.savefig(Save_score_figure, bbox_inches='tight')
