# 原始交易策略模块 - 保存原始策略作为备份
# 策略关键点：按照信号开仓，如果1开多，-1开空，0不开仓；按照持有3个bar就平仓；
# 如果持仓中继续有1或者-1，则延长；如果多仓出现-1信号立刻平仓

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class SignalBasedTradingStrategy:
    """
    基于信号的交易策略类（原始策略）
    实现用户要求的交易规则：
    1. 按信号开仓（1开多，-1开空，0不开仓）
    2. 持有3个bar平仓
    3. 持仓中相同信号延长持仓
    4. 反向信号立即平仓
    5. 固定1手开仓
    6. 固定10个价位止盈
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
        self.position_enter_price = 0  # 记录开仓价格
        self.position_history = []  # 记录所有交易
        self.current_bar = 0  # 当前bar计数
    
    def get_trade_signal(self, signal: int, current_index: int, price_info: Dict = None) -> Tuple[int, str]:
        """
        根据当前信号和持仓状态生成交易信号
        
        参数:
        signal: 当前预测信号 (1: 多头, -1: 空头, 0: 无信号)
        current_index: 当前数据索引
        price_info: 包含当前价格信息的字典，必须包含 'close', 'high', 'low' 键
        
        返回:
        Tuple[int, str]: (交易动作, 说明)
        交易动作: 0: 无操作, 1: 开多, -1: 开空, 2: 平多, -2: 平空
        """
        trade_action = 0
        explanation = "无操作"
        
        # 策略逻辑实现
        if self.current_position == 0:  # 空仓状态
            if signal == 1 and price_info is not None:
                # 开多仓
                trade_action = 1
                explanation = "开多仓（固定1手）"
                self.current_position = 1
                self.position_enter_time = current_index
                self.position_enter_price = price_info['close']
            elif signal == -1 and price_info is not None:
                # 开空仓
                trade_action = -1
                explanation = "开空仓（固定1手）"
                self.current_position = -1
                self.position_enter_time = current_index
                self.position_enter_price = price_info['close']
        elif self.current_position == 1:  # 多仓状态
            # 检查是否达到止盈条件
            if price_info is not None and price_info['high'] >= self.position_enter_price + 10:
                trade_action = 2
                explanation = "止盈平仓（上涨10个价位）"
                self._record_trade('long', self.position_enter_time, current_index, 
                                  self.position_enter_price, price_info['high'])
                self.current_position = 0
                self.position_enter_price = 0
            # 反向信号立即平仓
            elif signal == -1:
                trade_action = 2
                explanation = "反向信号平仓（多仓遇到空头信号）"
                if price_info is not None:
                    self._record_trade('long', self.position_enter_time, current_index, 
                                      self.position_enter_price, price_info['close'])
                else:
                    self._record_trade('long', self.position_enter_time, current_index)
                self.current_position = 0
                self.position_enter_price = 0
            # 相同信号延长持仓时间
            elif signal == 1:
                explanation = "延长持仓时间（多仓遇到多头信号）"
                self.position_enter_time = current_index  # 重置持仓时间
            # 持仓时间到达后平仓
            elif current_index - self.position_enter_time >= self.lookahead:
                trade_action = 2
                explanation = f"持仓时间到达平仓（持有{current_index - self.position_enter_time}个bar）"
                if price_info is not None:
                    self._record_trade('long', self.position_enter_time, current_index, 
                                      self.position_enter_price, price_info['close'])
                else:
                    self._record_trade('long', self.position_enter_time, current_index)
                self.current_position = 0
                self.position_enter_price = 0
        elif self.current_position == -1:  # 空仓状态
            # 检查是否达到止盈条件
            if price_info is not None and price_info['low'] <= self.position_enter_price - 10:
                trade_action = -2
                explanation = "止盈平仓（下跌10个价位）"
                self._record_trade('short', self.position_enter_time, current_index, 
                                  self.position_enter_price, price_info['low'])
                self.current_position = 0
                self.position_enter_price = 0
            # 反向信号立即平仓
            elif signal == 1:
                trade_action = -2
                explanation = "反向信号平仓（空仓遇到多头信号）"
                if price_info is not None:
                    self._record_trade('short', self.position_enter_time, current_index, 
                                      self.position_enter_price, price_info['close'])
                else:
                    self._record_trade('short', self.position_enter_time, current_index)
                self.current_position = 0
                self.position_enter_price = 0
            # 相同信号延长持仓时间
            elif signal == -1:
                explanation = "延长持仓时间（空仓遇到空头信号）"
                self.position_enter_time = current_index  # 重置持仓时间
            # 持仓时间到达后平仓
            elif current_index - self.position_enter_time >= self.lookahead:
                trade_action = -2
                explanation = f"持仓时间到达平仓（持有{current_index - self.position_enter_time}个bar）"
                if price_info is not None:
                    self._record_trade('short', self.position_enter_time, current_index, 
                                      self.position_enter_price, price_info['close'])
                else:
                    self._record_trade('short', self.position_enter_time, current_index)
                self.current_position = 0
                self.position_enter_price = 0
        
        self.current_bar = current_index
        return trade_action, explanation
    
    def _record_trade(self, trade_type: str, entry_index: int, exit_index: int, 
                     entry_price: float = None, exit_price: float = None):
        """
        记录交易历史
        
        参数:
        trade_type: 交易类型 ('long' 或 'short')
        entry_index: 入场索引
        exit_index: 出场索引
        entry_price: 入场价格（可选）
        exit_price: 出场价格（可选）
        """
        trade_record = {
            'type': trade_type,
            'entry_index': entry_index,
            'exit_index': exit_index,
            'position_size': 1,  # 固定1手
            'holding_period': exit_index - entry_index + 1  # 持有bar数量
        }
        
        if entry_price is not None:
            trade_record['entry_price'] = entry_price
        if exit_price is not None:
            trade_record['exit_price'] = exit_price
            # 计算盈亏
            if trade_type == 'long':
                trade_record['pnl'] = (exit_price - entry_price) * 1  # 固定1手
            elif trade_type == 'short':
                trade_record['pnl'] = (entry_price - exit_price) * 1  # 固定1手
        
        self.position_history.append(trade_record)
    
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
        self.position_enter_price = 0
        self.position_history = []
        self.current_bar = 0
        
    def close_position(self):
        """
        关闭当前持仓
        
        返回:
        bool: 是否成功平仓
        """
        if self.current_position != 0:
            # 记录平仓历史
            trade_type = 'long' if self.current_position == 1 else 'short'
            self._record_trade(trade_type, self.position_enter_time, self.current_bar)
            self.current_position = 0
            self.position_enter_price = 0
            return True
        return False

# 策略工厂函数，用于动态加载原始策略
def create_original_strategy(strategy_config: Dict) -> SignalBasedTradingStrategy:
    """
    创建原始交易策略实例
    
    参数:
    strategy_config: 策略配置参数
    
    返回:
    SignalBasedTradingStrategy: 交易策略实例
    """
    lookahead = strategy_config.get('lookahead', 3)
    return SignalBasedTradingStrategy(lookahead=lookahead)