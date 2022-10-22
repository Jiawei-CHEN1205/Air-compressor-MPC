# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:59:41 2022

@author: zhangkaiyuan
"""
import json
import redis
import threading
from threading import Thread
import requests
import time
import uuid
import pandas as pd
import numpy as np
from mq_http_sdk.mq_exception import MQExceptionBase
from commons.mq import MQProducer
from commons.reminder import send_robot_message
from control.control import controler
from commons.utils import get_logger, trans_date, read_yaml
from commons.dataprocessing import get_config, MysqlConnect

logger = get_logger(__file__,
                    log_dir=f'/logs/air_compressor_mpc_{threading.currentThread().ident}.log')


class ControlTiming(Thread):
    def __init__(self,
                 config,
                 station):
        super().__init__()
        self.station_name = station['name']
        self.enterprise_id = station['enterprise_id']
        self.device_group_id = station['device_group_id']
        self.station = station
        self.mysql_config = config['mysql']
        logger.info(json.dumps(dict(device_group_id=station['device_group_id'], algorithmId=station['algorithm_id']),
                               ensure_ascii=False))
        # 配置云智控redis库
        cic_redis_params = config['cic_redis']
        self.cic_redis = redis.Redis(host=cic_redis_params['host'],
                                     port=cic_redis_params['port'],
                                     password=cic_redis_params['key'],
                                     db=cic_redis_params['db'],
                                     decode_responses=True)

        # 配置算法redis库
        algo_redis_params = config['algo_redis']
        self.algo_redis = redis.Redis(host=algo_redis_params['host'],
                                      port=algo_redis_params['port'],
                                      password=algo_redis_params['key'],
                                      db=algo_redis_params['db'],
                                      decode_responses=True)
        # 初始化MQ
        self.producer = MQProducer(config['send_mq'])
        # 获取边缘端压力上下限及预警值
        url = config['edge_param']
        header = {"Content-Type": "application/json"}
        post_json = json.dumps({"groupId": station['device_group_id']})
        response = requests.post(url, data=post_json, headers=header).json()
        if response['code'] == 200:
            edge_data = response['data']
            self.pressure_threshold = self.get_edge(edge_data)
            self.joint_algo_id = edge_data['jointAlgoId']
        else:
            self.pressure_threshold = None
            self.joint_algo_id = ''
        logger.info(json.dumps(dict(pressure_threshold=self.pressure_threshold, joint_algo_id=self.joint_algo_id),
                               ensure_ascii=False))
        # 参数更新时间
        self.edge_time = int(time.time() * 1000)
        # 获取待机队列
        self.awaited_url =\
            config['queue_url'] + f"algorithmId={station['algorithm_id']}&groupId={station['device_group_id']}&type=0"
        # 获取运行队列
        self.running_url = \
            config['queue_url'] + f"algorithmId={station['algorithm_id']}&groupId={station['device_group_id']}&type=1"
        # 定义MQ消息的格式
        self.message = {
            'requestId': '',
            'type': 4,
            'aiReq': {
                'type': 0,  # 开机:1，关机:2
                'groupId': station['device_group_id'],
                'algorithmId': station['algorithm_id'],
                'queue': [],
                'reason': '',
            }
        }
        # 是否向钉钉机器人发送消息
        self.robot = station['robot_message']
        # 是否发送控制信息
        self.control_message = station['control_message']
        # 控制算法开关, 从redis获取, 只有值为'1'时才发送控制信息
        self.switch = None

    @staticmethod
    def get_edge(res):
        end_press = res['endPress']
        press_drop = res['pressDrop']
        safe_press = res['safePress']
        lower_limit = end_press + press_drop + safe_press
        low_warn = res.get('lowWarnPress', None)
        up_warn = res.get('upWarnPress', None)
        up_limit = res['upLimitPress']
        if any((lower_limit, low_warn, up_warn, up_limit)):
            return lower_limit, low_warn, up_warn, up_limit      
        
    def run(self):
        param = {'param_pressure':{'code':self.station['pressure']['code'],
                                   'equipment_id':self.station['pressure']['equipment_id']},
                 'param_power':{i['power_id']:{'code':i['power_code'],
                                               'equipment_id':i['power_id']} 
                                for i in self.station['compressors']},
                 'param_status':{i['equipment_id']:{'code':i['runstatus_code'],
                                               'equipment_id':i['equipment_id']} 
                                for i in self.station['compressors']}}
        ignore_machine = [i['equipment_id'] for i in self.station['compressors'] if i['ignore']]
        control_param = {'penatly_down':self.station['penatly_down'],
                         'penatly_up':self.station['penatly_up'],
                         'ignore_machine':ignore_machine,
                         'up_fraction':self.station['up_fraction'],
                         'down_fraction':self.station['down_fraction'],
                         'up_raise':self.station['up_raise'],
                         'down_raise':self.station['down_raise'],
                         'down_steps':self.station['down_steps'],
                         'up_steps':self.station['up_steps']}
        enterprise_id = self.enterprise_id 
        group_id = self.device_group_id
        down_pressure = self.pressure_threshold[0]
        up_pressure = self.pressure_threshold[3]
        goal_pressure = down_pressure
        dic_air_power = {i['equipment_id']:i['power_id'] for i in self.station['compressors']}
        control = controler(enterprise_id, 
                            group_id, 
                            dic_air_power, 
                            goal_pressure, 
                            down_pressure, 
                            up_pressure,
                            control_param)        
        while 1:            
            if int(time.time())%10==0:
                try:
                    # 算法开关
                    self.switch = self.algo_redis.get(f'AI:ShortTermSwitch:{self.device_group_id}')
                    if self.switch=='0':
                        time.sleep(10)
                        continue
                except Exception as e:
                    logger.error(f'switch error:{e}')
                    time.sleep(10)
                    continue
                # 判断控制参数是否更新
                try:
                    cache_data = self.algo_redis.get(f"AI:EdgeParam:{self.joint_algo_id}")
                    if cache_data:
                        cache = json.loads(str(cache_data))
                        if cache['born_timestamp'] > self.edge_time:
                            self.pressure_threshold = self.get_edge(cache)
                            self.edge_time = cache['born_timestamp']
                            logger.info(f'The pressure threshold has been changed: {self.pressure_threshold}')
                            # 变更控制参数
                            middle_pressure = (self.pressure_threshold[0]+self.pressure_threshold[-1])/2
                            control.down_pressure = self.pressure_threshold[0]
                            control.up_pressure = self.pressure_threshold[3]
                            control.middle_pressure = middle_pressure
                except Exception as e:
                    logger.error(f'Update pressure threshold Exception:{e}')
                # 获取加卸载队列
                try:
                    res = requests.get(self.awaited_url).json()
                    awaited_queue = res['data']
                    loadlist = [i['equipmentId'] for i in awaited_queue]
                    res = requests.get(self.running_url).json()
                    running_queue = res['data']
                    closelist = [i['equipmentId'] for i in running_queue]
                except Exception as e:
                    logger.error(f'Getting awaited queue failed: {e}')
                    loadlist, closelist = [],[]
                # MPC控制
                try:
                    machine, operate, info = control.rolling(param, loadlist=loadlist, closelist=closelist)
                    if machine:
                        operate_dic = {1:'开机', -1:'关机'}
                        pressure = info['pressure']
                        instruct = operate_dic[operate]
                        msg = "%s, 提前%s"%(self.station_name, instruct)
                        queue = info['operate']
                        # dingding群信息发送
                        if self.robot:
                            robot_message = (trans_date(int(time.time())),
                                                         instruct,
                                                         pressure,
                                                         msg,
                                                         queue)
                            send_robot_message(robot_message)
                        else:
                            judgement = dict(robot=self.robot,
                                             instruct=instruct,
                                             switch=self.switch)
                            logger.warn(f'No sending robot: {judgement}')                        
                        # 控制信息发送             
                        if self.control_message:
                            request_id = str(uuid.uuid1())
                            self.message['requestId'] = request_id.replace('-', '')
                            self.message['aiReq']['type'] = 1 if instruct == '开机' else 2
                            self.message['aiReq']['reason'] = msg
                            self.message['aiReq']['queue'] = [{"equipmentId":machine}]
                            try:
                                self.producer.send(self.message)
                                logger.info(f"Publish message succeed. Message:{self.message}")
                            except MQExceptionBase as e:
                                logger.error("Publish message fail. Exception:%s" % e)
                        else:
                            judgement = dict(control_message=self.control_message,
                                             instruct=instruct,
                                             switch=self.switch)
                            logger.warn(f'No sending control message: {judgement}')    
                        # mysql入库
                        #try:
                        mysql_info = {'timestamp':int(time.time()),
                                      'timestring':trans_date(int(time.time())),
                                      'enterprise_id':self.enterprise_id,
                                      'device_group_id':self.device_group_id,
                                      'pressure':pressure,
                                      'loadlist':'|'.join(loadlist),
                                      'closelist':'|'.join(closelist),
                                      'operateinfo':queue}  
                        self.mysql_connect = MysqlConnect(self.mysql_config)
                        self.mysql_connect.save_data(pd.DataFrame([mysql_info]), 'mpc_info')
                        self.mysql_connect.close_con()
                except Exception as e:
                    logger.error(f'mpc control failed: {e}')
                    time.sleep(2)
                    continue
                time.sleep(2)
            else:
                time.sleep(1)


if __name__=='__main__':
    config_name = get_config('test')
    config = read_yaml(config_name)
    for station in config['air_stations_config']:
        station_config = read_yaml(station['config_name'])
        timing = ControlTiming(config=config, station=station_config)
        timing.main()
    queue = '5#空压机开机'
    pressure = 6.8
    loadlist = ['7f89b1ac00104359b61557f0cc591144', '9083622e2ebe4e6db641bf10364dab13', '4f9bf1d26a3a41b7819aba6894be2944'] 
    closelist = ['3ccd7c0289644e33823d22ec96072ed5']
