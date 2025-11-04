# 交易策略模块 - 按照用户需求实现的交易策略
# 策略关键点：按照信号开仓，如果1开多，-1开空，0不开仓；按照持有3个bar就平仓；
# 如果持仓中继续有1或者-1，则延长；如果多仓出现-1信号立刻平仓

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class SignalBasedTradingStrategy:
    """
    基于信号的交易策略类
    实现用户要求的交易规则：
    1. 按信号开仓（1开多，-1开空，0不开仓）
    2. 持有3个bar平仓
    3. 持仓中相同信号延长持仓
    4. 反向信号立即平仓
    """
    
    def __init__(self, lookahead: int = 3):
        """
        初始化交易策略
        
        参数:
        lookahead: 目标持有时间（bar数量），默认为3
        """
        self.lookahead = lookahead
        self.current_position = 0  # 0: 空仓, 1: 多仓, -1: 空仓
        self.position_enter_time = 0  # 持仓进入时间戳或索引
        self.position_history = []  # 记录所有交易
    
    def get_trade_signal(self, signal: int, current_index: int) -> Tuple[int, str]:
        """
        根据当前信号和持仓状态生成交易信号
        
        参数:
        signal: 当前预测信号 (1: 多头, -1: 空头, 0: 无信号)
        current_index: 当前数据索引
        
        返回:
        Tuple[int, str]: (交易动作, 说明)
        交易动作: 0: 无操作, 1: 开多, -1: 开空, 2: 平多, -2: 平空
        """
        trade_action = 0
        explanation = "无操作"
        
        # 检查是否到达强制平仓时间（lookahead）
        force_close = False
        if self.current_position != 0:
            if current_index - self.position_enter_time >= self.lookahead:
                force_close = True
        
        # 策略逻辑实现
        if self.current_position == 0:  # 空仓状态
            if signal == 1:
                # 开多仓
                trade_action = 1
                explanation = "开多仓"
                self.current_position = 1
                self.position_enter_time = current_index
            elif signal == -1:
                # 开空仓
                trade_action = -1
                explanation = "开空仓"
                self.current_position = -1
                self.position_enter_time = current_index
        elif self.current_position == 1:  # 多仓状态
            if signal == -1 or force_close:
                # 反向信号或强制平仓，平多仓
                trade_action = 2
                explanation = "反向信号或强制平仓，平多仓"
                self._record_trade('long', self.position_enter_time, current_index)
                self.current_position = 0
            elif signal == 1:
                # 相同信号，延长持仓
                explanation = "相同多头信号，延长持仓"
                # 重新计时
                self.position_enter_time = current_index
            else:  # signal == 0
                # 无信号，继续持仓，不延长
                explanation = "无信号，继续持仓"
        elif self.current_position == -1:  # 空仓状态
            if signal == 1 or force_close:
                # 反向信号或强制平仓，平空仓
                trade_action = -2
                explanation = "反向信号或强制平仓，平空仓"
                self._record_trade('short', self.position_enter_time, current_index)
                self.current_position = 0
            elif signal == -1:
                # 相同信号，延长持仓
                explanation = "相同空头信号，延长持仓"
                # 重新计时
                self.position_enter_time = current_index
            else:  # signal == 0
                # 无信号，继续持仓，不延长
                explanation = "无信号，继续持仓"
        
        return trade_action, explanation
    
    def _record_trade(self, trade_type: str, entry_index: int, exit_index: int):
        """
        记录交易历史
        
        参数:
        trade_type: 交易类型 ('long' 或 'short')
        entry_index: 入场索引
        exit_index: 出场索引
        """
        self.position_history.append({
            'type': trade_type,
            'entry_index': entry_index,
            'exit_index': exit_index
        })
    
    def get_position_history(self) -> List[Dict]:
        """
        获取所有交易历史
        
        返回:
        List[Dict]: 交易历史列表
        """
        return self.position_history
    
    def reset(self):
        """
        重置策略状态
        """
        self.current_position = 0
        self.position_enter_time = 0
        self.position_history = []
        
    def close_position(self):
        """
        关闭当前持仓
        
        返回:
        bool: 是否成功平仓
        """
        if self.current_position != 0:
            # 记录平仓历史
            trade_type = 'long' if self.current_position == 1 else 'short'
            self._record_trade(trade_type, self.position_enter_time, self.position_enter_time)  # 简化记录
            self.current_position = 0
            return True
        return False

# 策略工厂函数，用于动态加载策略
def create_strategy(strategy_config: Dict) -> SignalBasedTradingStrategy:
    """
    创建交易策略实例
    
    参数:
    strategy_config: 策略配置参数
    
    返回:
    SignalBasedTradingStrategy: 交易策略实例
    """
    lookahead = strategy_config.get('lookahead', 3)
    return SignalBasedTradingStrategy(lookahead=lookahead)