# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:43:13 2022

@author: zhangkaiyuan
"""
import json
import redis
from threading import Thread
from commons.mq import MQConsumer
from commons.utils import get_logger

logger = get_logger(__file__, log_dir=f'/logs/edge_param_update.log')


class EdgeParamUpdate(Thread):

    def __init__(self, receive_mq, redis_params):

        super().__init__()
        self.consumer = MQConsumer(receive_mq)
        self.redis = redis.Redis(host=redis_params['host'],
                                 port=redis_params['port'],
                                 password=redis_params['key'],
                                 db=redis_params['db'],
                                 decode_responses=True)
        try:
            msg_mq = None
            for msg_data in self.consumer.consumer_message():
                msg = json.loads(str(msg_data))
                cache_data = self.redis.get(f"AI:EdgeParam:{msg['jointAlgoId']}")
                cache = None
                if cache_data:
                    cache = json.loads(str(cache_data))
                if cache is None or msg_data.born_timestamp > cache['born_timestamp'] and msg['jointAlgoId'] == cache['jointAlgoId']:
                    msg_mq = json.dumps(dict(endPress=msg['endPress'],
                                             pressDrop=msg['pressDrop'],
                                             safePress=msg['safePress'],
                                             lowWarnPress=msg['lowWarnPress'],
                                             upWarnPress=msg['upWarnPress'],
                                             upLimitPress=msg['upLimitPress'],
                                             jointAlgoId=msg['jointAlgoId'],
                                             born_timestamp=msg_data.born_timestamp))
                    self.redis.set(f"AI:EdgeParam:{msg['jointAlgoId']}", msg_mq)
                    logger.info(f'Init Redis pressure threshold: {msg_mq}')
        except Exception as e:
            logger.error(f'Consumer Exception:{e}')

    def run(self):
        while True:
            # mq消费，确定边缘智控参数是否有变化,如有变化，则更新参数
            try:
                msg_mq = None
                for msg_data in self.consumer.consumer_message():
                    msg = json.loads(str(msg_data))
                    cache_data = self.redis.get(f"AI:EdgeParam:{msg['jointAlgoId']}")
                    cache = None
                    if cache_data:
                        cache = json.loads(str(cache_data))
                    if cache is None or msg_data.born_timestamp > cache['born_timestamp'] and msg['jointAlgoId'] == cache['jointAlgoId']:
                        msg_mq = json.dumps(dict(endPress=msg['endPress'],
                                                 pressDrop=msg['pressDrop'],
                                                 safePress=msg['safePress'],
                                                 lowWarnPress=msg['lowWarnPress'],
                                                 upWarnPress=msg['upWarnPress'],
                                                 upLimitPress=msg['upLimitPress'],
                                                 jointAlgoId=msg['jointAlgoId'],
                                                 born_timestamp=msg_data.born_timestamp))
                        self.redis.set(f"AI:EdgeParam:{msg['jointAlgoId']}", msg_mq)
                        logger.info(f'Update Redis pressure threshold: {msg_mq}')
            except Exception as e:
                logger.error(f'Consumer Exception:{e}')

