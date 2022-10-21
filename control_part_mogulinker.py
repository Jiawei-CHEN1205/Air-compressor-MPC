# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:08:58 2022

@author: zhangkaiyuan
"""
import tensorflow as tf
import time
from commons.dataprocessing import data_process
from control.gru import weight_loss
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def combination(n, c, per=[], actions=2):
    """
    n:输入的id列表
    c:排列组合的长度
    actions:非0的id个数（其他都为0）
    """
    for inx, pos in enumerate(n):
        t = per + [pos]
        count_ = [ti for ti in t if ti!=0]
        if len(set(count_)) == len(count_) and len(count_)<=actions:
            if len(t) == c:
                yield [pos,]
            else:
                for result in combination(n, c, per + [pos,], actions):
                    yield [pos,] + result
class controler():
    def __init__(self,
                 enterprise_id, 
                 group_id, 
                 dic_air_power,
                 goal_pressure, 
                 down_pressure,
                 up_pressure,
                 control_param):
        # 模型初始化
        self.predicter = tf.keras.models.load_model('./model/%s'%group_id, custom_objects={'weight_loss':weight_loss})
        # 目标压力
        self.goal_pressure = goal_pressure
        # 下限压力
        self.down_pressure = down_pressure + control_param['up_raise']
        # 上限压力
        self.up_pressure = up_pressure - control_param['down_raise']
        # 控制中线压力
        self.middle_pressure = (down_pressure+up_pressure)/2
        # 控制参数
        self.control_param = control_param
        self.params = self.get_params(group_id)
        self.config_power_low = {k:self.params[k] for k in self.params.keys() if k.endswith('_close')}
        # 电表-空压机映射表
        self.dic_air_power = dic_air_power
        self.dic_power_air = {v:k for k,v in dic_air_power.items()}
        # 动作列表
        self.status_list = [v+'_status' for v in self.dic_air_power.keys()]
        # 变频字典
        self.conversion_dic = {k.split('_')[0]:self.config_power_low[k] for k in self.config_power_low.keys() if self.config_power_low[k]!=1}
        self.conversion_list = [self.dic_power_air[k] for k in self.conversion_dic.keys()]
        # 数据读取初始化
        self.dataloader = data_process(enterprise_id, group_id)
        self.air_id = sorted(self.dataloader.get_air_list.values())
        self.air_dic = dict(zip(self.dataloader.get_air_list.values(),self.dataloader.get_air_list.keys()))
        self.tokens_open = dict(zip(self.air_id,np.diag([1] * len(self.air_id))))
        self.tokens_close = dict(zip(self.air_id,np.diag([-1] * len(self.air_id))))
        self.token_0 = {0:np.zeros((1,len(self.air_id)))}
        # 动作初始化，假设过去步长没有开关机,固定窗口为6，后面可以改为可配置。
        self.u = np.zeros((6, len(self.air_id)))
        self.operation = {'machine':0,'operate':0, 'index':0}
        # 获取实时状态数据
        # self.s = self.get_s_data()
        self.fivepass = 0
    
    def get_operates(self, loadlist, closelist, pressure_now, slope_now, actions=3):
        tokens = self.token_0.copy()
        # 后续可以考虑按照超过目标压力只关机
        if pressure_now<self.middle_pressure and slope_now<0:
            open_list = {id_key:id_val for id_key, id_val 
                        in self.tokens_open.items() if id_key in loadlist}
            tokens.update(open_list)
        elif pressure_now>=self.middle_pressure and slope_now>0:
            close_list = {id_key:id_val for id_key, id_val 
                        in self.tokens_close.items() if id_key in closelist}
            tokens.update(close_list)
        id_ = []
        operates = []
        for ans in combination(tokens.keys(), 6, actions=actions):
            u = np.zeros((6, len(self.air_id)))
            for inx, i in enumerate(ans):
                u[inx, :] = tokens[i] 
            operates.append(u)
            id_.append(ans)
        return np.array(operates), id_

        
    def get_queue(self, 
                  param, 
                  real_data, 
                  start_time = None, 
                  end_time = None):
        if not start_time:
            start_time = int(time.time()*1e3-120*1e3)
            end_time = int(time.time()*1e3)
        loadlist,closelist,loadlist_conversion,unloadlist = [],[],[],[]
        for inx, id_ in enumerate(self.air_id):
            # 看之前几步是否有操作
            if np.sum(self.u[:, inx])!=0:
                continue       
            # 看是否处于低频,需要空压机和电表映射
            power_col= self.dic_air_power[id_]
            power_now = real_data[power_col+'_power'].values[-1]
            param_ = param['param_status'][id_]
            run_ = self.dataloader.get_runstatus(start_time, end_time, param_)
            # 美特5#机自行加卸载，开机时段先不发送消息
            if id_ in self.control_param['ignore_machine']:
                if run_['value'].values[-1] in ['load', 'run', 'unload']:
                    self.fivepass = 1
                else:
                    self.fivepass = 0 
            if run_['value'].values[-1] in ['load', 'run'] and power_now<=self.config_power_low[power_col+'_close']:
                closelist.append(id_)
            if run_['value'].values[-1] in ['unload', 'stop'] and id_ not in self.conversion_list:
                loadlist.append(id_)
            if run_['value'].values[-1] in ['unload', 'stop'] and id_ in self.conversion_list:
                loadlist_conversion.append(id_)
            #TODO: 卸载机器优先加载？
            if run_['value'].values[-1] in ['unload']:
                unloadlist.append(id_)
        if len(loadlist_conversion)>0:
            loadlist = loadlist_conversion
        return loadlist, closelist, unloadlist
    
    
    def get_s_data(self, param):
        s = self.dataloader.main(param, train=0)
        return s.drop(columns=['timestamp', 'slope_slope'])[-6:]
        
     
    def get_input(self, s, all_u, u):
        """
        s:状态矩阵
        all_u:历史动作矩阵
        u:最新动作
        """
        all_u = tf.concat([all_u, u], 1)[:,1:,:]
        input_ = tf.concat([s, all_u], -1)
        return all_u, input_
    
    def get_params(self, group_id):
        with open('./config/%s.json'%group_id,'r') as load_f:
            config_power = json.load(load_f)
        return config_power
    
    def predict(self, s, u):
        nums = u.shape[0]
        feature_len = s.shape[-1]
        all_s = tf.cast(np.array([s for i in range(nums)]),dtype='float32')
        all_u = tf.cast(np.array([self.u for i in range(nums)]),dtype='float32')
        for i in range(u.shape[1]):
            if i<u.shape[1]:
                all_u, input_ = self.get_input(all_s, all_u, u[:,i,:].reshape(nums,1,u.shape[-1]))
            else:
                all_u, input_ = self.get_input(all_s, all_u, np.zeros(u[:,0,:].shape).reshape(nums,1,u.shape[-1]))
            pred = self.predicter(input_)
            pressure = pred[:,0] + all_s[:,-1,0]
            slope_30 = pressure - all_s[:,-3,0]
            pred = np.insert(pred, 0, values=pressure, axis=1)
            pred = np.insert(pred, 2, values=slope_30, axis=1)
            # pred[:,2] = slope_30
            if pred.shape[-1]!=feature_len:
                print("滚动预测输入维度错误，请检查")
            all_s = tf.concat([all_s[:,1:,:], pred.reshape(nums,1,feature_len)], 1)
        return all_s, all_u
    
    def static(self):
        self.operation['machine'] = 0
        self.operation['operate'] = 0  
        self.operation['index'] = 0  
        
    def loss(self, all_s, all_u, pressure_now):

        weight_len = 4
        # 约束条件1：所有压力不能低于下限
        # 约束条件2：开机的机器历史一小时次数不能超过2
        if pressure_now>self.middle_pressure:
            min_u = self.middle_pressure+(self.up_pressure-self.middle_pressure)*self.control_param['up_fraction']
            if pressure_now>=min_u:
                weight_u = 0.8*abs(pressure_now-min_u)/(self.up_pressure-min_u)
            else:
                weight_u = 2*abs(pressure_now-min_u)/(min_u-self.middle_pressure)
            
        else:
            min_u = self.down_pressure+(self.middle_pressure-self.down_pressure)*self.control_param['down_fraction']
            if pressure_now>=min_u:
                weight_u = 1.5*abs(pressure_now-min_u)/(self.middle_pressure-min_u)
            else:
                weight_u = 0.8*abs(pressure_now-min_u)/(min_u-self.down_pressure)            
        # weight_u = abs(pressure_now-min_u)/((self.up_pressure-self.down_pressure)/4)
        #weight_u = 1-abs(pressure_now-self.middle_pressure)/((self.up_pressure-self.down_pressure)/2)
        if pressure_now>=self.middle_pressure:
            weight = [30+weight_u*200,10,13000,1]       
            loss_slope = np.sum((all_s[:,:2,1]+0.001)**2, axis=1)
        else:
            weight = [50+weight_u*200,10,14000,1] 
            loss_slope = np.sum((all_s[:,weight_len:,1]-0.001)**2, axis=1)
        loss_u = np.power(np.sum(all_u**2, axis=(1,2)),1/5)
        loss_pressure = np.sum((all_s[:,weight_len-2:weight_len,0]-self.goal_pressure)**2, axis=1)
        # loss_pressure_low = np.sum((all_s[:,weight_len-2:weight_len,0]-self.down_pressure)**2, axis=1)

        loss_unload = np.sum((all_s[:,weight_len,3:]>0.2) & (all_s[:,weight_len,3:]<0.6),axis=1)
        loss_ = np.average([loss_u, loss_pressure, loss_slope, loss_unload],weights=weight,axis=0)
        # 压力下限
        loss_[np.sum((all_s[:,:self.control_param['down_steps'],0]<=self.down_pressure), axis=1)>0] *= self.control_param['penatly_down']
        # 压力上限
        loss_[np.sum((all_s[:,:self.control_param['up_steps'],0]>=self.up_pressure), axis=1)>0] *= self.control_param['penatly_up']
        return list(loss_)
    
    def rolling(self, param, loadlist=None, closelist=None):
        # 获取状态s
        real_data = self.get_s_data(param)
        s = real_data.drop(columns=self.status_list).values
        self.u = real_data[self.status_list].values
        pressure_now = s[-1,0]
        slope_now = s[-1, 1]
        info = {'timestamp':time.time(), 
                'pressure':pressure_now, 
                'slope':s[-1,1], 
                'slope_30':s[-1, 2],
                'details':'执行成功'}
        if self.operation['operate']!=0 and self.u[-1,self.operation['index']]==0 and self.operation['machine'] not in loadlist+closelist:
            info['details'] = '上步指令锁定，且状态没变'
            return 0, 0, info
        if pressure_now<=self.down_pressure or pressure_now>=self.up_pressure:
            info['details'] = '压力不满足要求'
            self.static()
            return 0, 0, info 
        if any(self.u[-1]==1) or self.operation['operate']==1:
            info['details'] = '前一步有开机'
            self.static()
            return 0, 0, info
        if any(self.u[-2]==1):
            info['details'] = '前两步有开机'
            self.static()
            return 0, 0, info
        if any(self.u[-3]==1):
            info['details'] = '前三步有开机'
            self.static()
            return 0, 0, info        
        if any(self.u[-1]==-1) or self.operation['operate']==-1:
            info['details'] = '前一步有关机'
            self.static()
            return 0, 0, info
        # 获取开关机队列
        loadlist_real, closelist_real, unloadlist = self.get_queue(param, real_data)
        if self.fivepass:
            self.static()
            info['details'] = '机器自行加卸载运行状态'
            return 0, 0, info            
        if not loadlist and not closelist:
            self.static()
            info['details'] = '待机队列和运行队列为空'
            return 0, 0, info
        else:
            set_ = (set(self.conversion_list)&set(loadlist))|(set(unloadlist)&set(loadlist))
            low_power = [self.dic_power_air[n] for n in self.conversion_dic.keys() if real_data[n+'_power'].values[-1]>self.conversion_dic[n]]
            if set_:
                loadlist = list(set_)
            for p in low_power:
                if p in closelist:
                    closelist.remove(p)
        info['loadlist'] = loadlist
        info['closelist'] = closelist        
        # 判断是否发送失败
        for inx,i in enumerate(self.u[-1, :]):
            if i==1 and self.air_id[inx] in loadlist:
                self.u[-1,inx] = 0
            if i==-1 and self.air_id[inx] in closelist:
                self.u[-1,inx] = 0
        # 获取全部可选择的u
        u, id_ = self.get_operates(loadlist, closelist, pressure_now, slope_now)
        # 预测
        all_s, all_u = self.predict(s, u)

        loss_ = self.loss(all_s, all_u, pressure_now)

        best_series_inx = loss_.index(min(loss_))
        if min(loss_)==np.inf:
            self.static()
            return 0,0,info
        best_u = all_u[best_series_inx, 0, :].numpy()
        best_s = all_s[best_series_inx, :, 0].numpy()
        info['predict'] = best_s
        info['loss_0'] = loss_[0]
        info['loss_min'] = min(loss_)

        for inx,id_ in enumerate(self.air_id):
            info['%s_power'%id_] = s[-1,2+inx]
        
        for inx, operate in enumerate(best_u):
            if operate != 0:
                info['operate'] = self.air_dic[self.air_id[inx]]+str(operate)
                self.operation['machine'] = self.air_id[inx]
                self.operation['operate'] = operate
                self.operation['index'] = inx
                return self.air_id[inx], operate, info
        info['operate'] = 0  
        self.static()
        return 0, 0, info

    
if __name__=='__main__':
    enterprise_id='e6231157d4ee4ca49685373db8670147'
    group_id='eb918b19f680405c9ca5592f64836a21'
    param = {'param_pressure':{'code':['pressure0_1'],'equipment_id':'7fed455ee1b3421cb8b1df7cad767b37'},
             'param_power':{'code':['totalactivePower'],'equipment_id':''},
             'param_status':{'code':['runStatusDetail'],'equipment_id':''}}
    dic_air_power = {'24854f5a21164839be72e6813093a435':'291ba0db7324470690e819a467bf71ee',
                 'bb236f5ef12e4f049ef5af92219336b2':'76c27402a4f34e9eb8229a914fdba2aa',
                 '9878b0e7f60a4e1c9cca6257688f0e70':'638327ecf1fb44d791f03e683d5508e8',
                 '12c480aae7794f4c942e203e08f465b7':'5b4618240b4742f5bf61bd224cccc41d',
                 '22b5f19a0ebb4a86b9efdb6592cd6b56':'679ceda4fefe4e7ebd6b1f40edc86a9b'}
    down_pressure = 6.4
    up_pressure = 7
    goal_pressure = 6.7
    control = controler(enterprise_id, group_id, dic_air_power, goal_pressure, down_pressure, up_pressure)