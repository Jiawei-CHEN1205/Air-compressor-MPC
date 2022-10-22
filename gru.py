# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:33:30 2022

@author: zhangkaiyuan
"""

import numpy as np
import warnings
import tensorflow as tf
from commons.dataprocessing import data_process
warnings.filterwarnings('ignore')
tf.random.set_seed(1234)

def weight_loss(y_true, y_pred):
    nums = y_true.shape[-1]-1
    weight = [300] + [1 for i in range(nums)]
    return tf.reduce_mean(tf.math.square(y_pred - y_true)*weight, axis=-1)

class model_train():
    def __init__(self,
                 enterprise_id, 
                 group_id,
                 windows):
        self.dataloader = data_process(enterprise_id,group_id)
        self.windows = windows
        
    def get_train(self, param):
        raw_data = self.dataloader.main(param, train=1)
        self.feature_list = ['timestamp', 'pressure', 'slope', 'slope_30'] + [i for i in raw_data.columns if 'power' in i] + [i for i in raw_data.columns if 'status' in i]
        self.target_list = ['timestamp', 'slope'] + [i for i in raw_data.columns if 'power' in i]
        train_num = len(raw_data) - self.windows
        train_all = np.zeros([train_num, self.windows, len(self.feature_list)])
        target_all = np.zeros([train_num, len(self.target_list)])
        # 训练数据生成
        operate_index = []
        change_index = []
        nochange_index = []
        for inx,ts in enumerate(raw_data['timestamp']):
            if inx>=train_num:
                break
            tmp = raw_data[(raw_data['timestamp']>=ts) & (raw_data['timestamp']<ts+self.windows)]
            if len(tmp[~tmp['operate'].isna()])>0 and len(tmp[tmp['change_out']==1])==0:
                operate_index.append(inx)
            elif len(tmp[~tmp['operate'].isna()])==0 and \
                 (raw_data[raw_data['timestamp']==ts+self.windows]['slope_slope'].values[0]>raw_data['slope_slope'].quantile(0.85)\
                      or raw_data[raw_data['timestamp']==ts+self.windows]['slope_slope'].values[0]<raw_data['slope_slope'].quantile(0.15)):
                change_index.append(inx)
            elif len(tmp[tmp['change_out']==1])>0:
                change_index.append(inx)
            elif raw_data.loc[inx,'change_tag']!=0:
                change_index.append(inx)
            else:
                nochange_index.append(inx)
            train_all[inx, :, :] = tmp[self.feature_list].values
            target_all[inx,:] = raw_data[raw_data['timestamp']==ts+self.windows][self.target_list].values
        self.raw_data = raw_data
        self.change_index = change_index
        self.operate_index = operate_index
        self.nochange_index = nochange_index
        return train_all, target_all
    
    def get_noise(self, train, target):
        # 增加压力噪音，随机调整压力位置
        train_all_noise = train.copy()
        target_all_noise = target.copy()
        needcount = train_all_noise.shape[0]
        noise = np.random.normal(0,1,needcount)
        train_all_noise[:,:, 1] = train_all_noise[:,:, 1]+np.array([noise for i in range(self.windows)]).T
        return train_all_noise, target_all_noise

    def get_rotate_target(self, rotate_train, rotate_target):
        # 压力曲线旋转
        noise = np.random.normal(0,2*self.raw_data['slope'].max(),rotate_train.shape[0],)
        rotate_train = rotate_train.copy()
        rotate_target = rotate_target.copy()
        # 30秒斜率加三倍噪音
        rotate_train[:,:,3] = rotate_train[:,:,3] + np.array([noise*3 for i in range(self.windows)]).T
        #rotate_train[:,:,2] = rotate_train[:,:,2] + np.array([noise*3 for i in range(windows)]).T
        # 斜率直接加噪音
        rotate_train[:,:,2] = rotate_train[:,:,2] + np.array([noise for i in range(self.windows)]).T
        rotate_target[:,1] = rotate_target[:,1] + noise
        # 压力要加步长的噪音
        rotate_train[:,:,1] = rotate_train[:,:,1] + np.array([noise*i for i in range(self.windows)]).T
        # rotate_target[:,1] = rotate_target[:,1] + noise*windows
        return rotate_train, rotate_target
    
    def input_(self, train_all, target_all, split_ratio = 0.8):
        
        split_point = int(len(train_all)*split_ratio)
        train_all_noise, target_all_noise = self.get_noise(train_all, target_all)
        X_train,X_test,y_train,y_test = train_all_noise[[inx for inx in self.nochange_index if inx<=split_point],:,:].copy(),\
                              train_all_noise[[inx for inx in self.nochange_index if inx>split_point],:,:].copy(),\
                              target_all_noise[[inx for inx in self.nochange_index if inx<=split_point],:].copy(),\
                              target_all_noise[[inx for inx in self.nochange_index if inx>split_point],:].copy()        
        # 重采样
        raw_train_ = train_all[[inx for inx in self.operate_index if inx<=split_point],:,:]
        raw_target_ = target_all[[inx for inx in self.operate_index if inx<=split_point],:]
        raw_test_ = train_all[[inx for inx in self.operate_index if inx>split_point],:,:]
        raw_target_test = target_all[[inx for inx in self.operate_index if inx>split_point],:]
        resample_ratio = int(0.5/(len(self.operate_index)/len(self.raw_data)))
        for i in range(resample_ratio):
            operate_train, operate_target = self.get_noise(raw_train_, raw_target_)
            rotate_train, rotate_target = self.get_rotate_target(operate_train, operate_target)
            operate_test, operate_target_test = self.get_noise(raw_test_, raw_target_test)
            rotate_test, rotate_target_test = self.get_rotate_target(operate_test, operate_target_test)    
            X_train = np.vstack([X_train]+[ operate_train, rotate_train])
            y_train = np.vstack([y_train]+[ operate_target, rotate_target])
            X_test = np.vstack([X_test]+[operate_test, rotate_test])
            y_test = np.vstack([y_test]+[operate_target_test, rotate_target_test])
        return X_train,y_train,X_test,y_test
    
    def gru_model(self, batch_size=256, output_size=None):
        if not output_size:
            output_size = len(self.target_list)-1
        model = tf.keras.Sequential()  #序列模型
        #model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.GRU(units = 64, 
                                      activation='tanh',
                                      return_sequences = True, 
                                      stateful=False)) 
    #     model.add(tf.keras.layers.Conv1D(32, 3, activation='relu'))
    #     model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    #     model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.GRU(units = 32, return_sequences = False, stateful=False))
        model.add(tf.keras.layers.Dense(units = 16)) 
        model.add(tf.keras.layers.Dense(units = output_size), ) 
        
        model.compile(loss = weight_loss, optimizer = tf.keras.optimizers.Nadam(0.0005), metrics =[weight_loss, tf.keras.metrics.MeanSquaredError()]) # 编译模型
        # 损失函数有： mean_squared_error， mean_absolute_error，huber_loss，log_cosh
        return model
    
    def train(self, batch_size):
        model = self.gru_model(batch_size)
        X_train,y_train,X_test,y_test = self.input_()
        # 提前停止训练，回调函数
        custom_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=4, # 容忍步数
            min_delta=0.00003,  # 最小调整量
            mode='min'
        )
        
        #保存模型，回调函数
        mcp_save = tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_best_only = True, monitor='val_loss', mode ='min')
        
        model.fit(X_train[:,:,1:], 
                            y_train[:,1:], 
                            epochs = 100, 
                            batch_size = batch_size,
                            #sample_weight=np.array(weight),
                            verbose = 0, 
                            validation_data=(X_test[:, :, 1:], y_test[:, 1:]), 
                            callbacks=[mcp_save, custom_early_stopping]) 
        
        test_sample = X_train[-1, :, 1:].copy()
        def get_sample(matrix, inx, new_value):
            ans = matrix.copy()
            ans[-1, -inx] = new_value
            return ans
        sample_open = np.array([get_sample(test_sample, i, 1) for i in range(1,len(self.target_list)-1)])
        sample_static = np.array([get_sample(test_sample, i, 0)  for i in range(1,len(self.target_list)-2)])
        sample_close = np.array([get_sample(test_sample, i, -1)  for i in range(1,len(self.target_list)-2)])
        open_pred, static_pred, close_pred = model.predict(sample_open), model.predict(sample_static), model.predict(sample_close)
        if any(open_pred<=static_pred) and any(static_pred<=close_pred):
            return "model is bad! please check data"
        else:
            model.save('./model/%s'%self.group_id)
            return "model has been trained"
    
if __name__=='__main__':
    param = {'param_pressure':{'code':['pressure0_1'],'equipment_id':'e4a18f770b914e18b64137b80f76538b'},
             'param_power':{'code':['totalactivePower'],'equipment_id':''},
             'param_status':{'code':['runStatusDetail'],'equipment_id':''}}
    enterprise_id='83f612293e924d70a7de1c3a967b82c4'
    group_id='1e90744bb13f495784756283e2acb26c'