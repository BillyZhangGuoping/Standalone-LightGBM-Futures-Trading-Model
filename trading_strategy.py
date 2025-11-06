# 交易策略模块 - 单方向多单策略实现
# 策略关键点：只开多单，持有3个bar强制平仓，平仓后50个bar冷静期

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class SingleLongOnlyStrategy:
    """
    单方向多单策略（使用lgbm_model_v1.0_251106_1643模型）
    实现规则：
    1. 只开多单，不允许开空单
    2. 持有3个bar后强制平仓，即使有新的多单信号也不延长
    3. 平仓后等待50个bar作为冷静期，期间不开仓
    4. 固定1手开仓
    """
    
    def __init__(self, holding_period: int = 3, cooldown_period: int = 50):
        """
        初始化单方向多单策略
        
        参数:
        holding_period: 强制持有期（bar数量），默认为3
        cooldown_period: 冷静期（bar数量），默认为50
        """
        self.holding_period = holding_period
        self.cooldown_period = cooldown_period
        self.current_position = 0  # 0: 空仓, 1: 多仓
        self.position_enter_time = 0  # 持仓进入时间戳或索引
        self.position_enter_price = 0  # 记录开仓价格
        self.position_history = []  # 记录所有交易
        self.last_exit_time = -cooldown_period  # 上次平仓时间，初始设为冷静期前
    
    def get_trade_signal(self, signal: int, current_index: int, price_info: Dict = None) -> Tuple[int, str]:
        """
        根据当前信号和持仓状态生成交易信号
        
        参数:
        signal: 当前预测信号 (1: 多头, -1: 空头, 0: 无信号)
        current_index: 当前数据索引
        price_info: 包含当前价格信息的字典，必须包含 'close', 'high', 'low' 键
        
        返回:
        Tuple[int, str]: (交易动作, 说明)
        交易动作: 0: 无操作, 1: 开多, 2: 平多
        """
        trade_action = 0
        explanation = "无操作"
        
        # 检查是否到达强制平仓时间
        force_close = False
        if self.current_position == 1:
            if current_index - self.position_enter_time >= self.holding_period:
                force_close = True
        
        # 策略逻辑实现
        if self.current_position == 0:  # 空仓状态
            # 检查是否在冷静期内
            in_cooldown = (current_index - self.last_exit_time) < self.cooldown_period
            
            if signal == 1 and price_info is not None and not in_cooldown:
                # 开多仓（只在非冷静期且有多头信号时开仓）
                trade_action = 1
                explanation = "开多仓（固定1手）"
                self.current_position = 1
                self.position_enter_time = current_index
                self.position_enter_price = price_info['close']
            elif in_cooldown:
                # 冷静期内，即使有信号也不开仓
                explanation = f"冷静期内（剩余{self.cooldown_period - (current_index - self.last_exit_time)}个bar），不开仓"
            else:
                # 无多头信号或价格信息不完整
                explanation = "无多头信号或价格信息不完整"
        elif self.current_position == 1:  # 多仓状态
            if force_close:
                # 强制平仓，无论后续是否有信号都平仓
                trade_action = 2
                explanation = "强制平仓（已持有3个bar）"
                if price_info is not None:
                    self._record_trade('long', self.position_enter_time, current_index, 
                                      self.position_enter_price, price_info['close'])
                else:
                    self._record_trade('long', self.position_enter_time, current_index)
                self.current_position = 0
                self.position_enter_price = 0
                self.last_exit_time = current_index  # 记录平仓时间，开始冷静期
        
        return trade_action, explanation
    
    def _record_trade(self, trade_type: str, entry_index: int, exit_index: int, 
                     entry_price: float = None, exit_price: float = None):
        """
        记录交易历史
        
        参数:
        trade_type: 交易类型 ('long')
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
            'model_used': 'lgbm_model_v1.0_251106_1643'  # 记录使用的模型
        }
        
        if entry_price is not None:
            trade_record['entry_price'] = entry_price
        if exit_price is not None:
            trade_record['exit_price'] = exit_price
            # 计算盈亏
            if trade_type == 'long':
                trade_record['pnl'] = (exit_price - entry_price) * 1  # 固定1手
        
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
        self.last_exit_time = -self.cooldown_period
        
    def close_position(self):
        """
        关闭当前持仓
        
        返回:
        bool: 是否成功平仓
        """
        if self.current_position != 0:
            # 记录平仓历史
            trade_type = 'long' if self.current_position == 1 else 'short'
            self._record_trade(trade_type, self.position_enter_time, self.position_enter_time)
            self.current_position = 0
            self.position_enter_price = 0
            self.last_exit_time = self.position_enter_time  # 更新最后平仓时间
            return True
        return False

# 策略工厂函数，用于动态加载策略
def create_strategy(strategy_config: Dict) -> object:
    """
    创建交易策略实例
    
    参数:
    strategy_config: 策略配置参数
    
    返回:
    交易策略实例
    """
    # 默认创建单方向多单策略
    holding_period = strategy_config.get('holding_period', 3)
    cooldown_period = strategy_config.get('cooldown_period', 50)
    return SingleLongOnlyStrategy(holding_period=holding_period, cooldown_period=cooldown_period)