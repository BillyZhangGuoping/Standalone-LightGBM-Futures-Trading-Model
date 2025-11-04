#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 回测模块

此模块负责模型预测结果的交易回测、性能分析和可视化
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from config import Config

# 设置日志
logger = logging.getLogger('backtester')


class TradingBacktester:
    """
    交易回测系统
    """
    
    def __init__(self):
        """
        初始化回测系统
        """
        # 使用getattr安全地获取Config属性
        self.initial_capital = getattr(Config, 'INITIAL_CAPITAL', 1000000.0)
        self.transaction_cost = getattr(Config, 'TRANSACTION_COST', 0.001)
        self.slippage = getattr(Config, 'SLIPPAGE', 0.0001)
        self.stop_loss = getattr(Config, 'STOP_LOSS', 0.0)
        self.take_profit = getattr(Config, 'TAKE_PROFIT', 0.0)
        self.max_position_size = getattr(Config, 'MAX_POSITION_SIZE', 1.0)
        self.evaluation_window = getattr(Config, 'EVALUATION_WINDOW', 20)
        self.backtest_results_dir = getattr(Config, 'BACKTEST_RESULTS_DIR', 'backtest_results')
        self.logger = logger
        
        # 回测结果
        self.backtest_results = None
        self.trade_history = []
        self.logger.info("交易回测系统初始化完成")
        
        # 确保回测结果目录存在
        os.makedirs(self.backtest_results_dir, exist_ok=True)
    
    def run_backtest(self, df: pd.DataFrame, pred_class: np.ndarray, 
                    pred_proba: np.ndarray, signal_threshold: float = 0.0) -> pd.DataFrame:
        """
        运行交易回测
        
        Args:
            df: 包含价格和预测的DataFrame
            pred_class: 预测类别
            pred_proba: 预测概率
            signal_threshold: 信号阈值，用于过滤弱信号
            
        Returns:
            pd.DataFrame: 回测结果
        """
        start_time = time.time()
        self.logger.info("开始交易回测")
        
        # 复制数据以避免修改原始数据
        backtest_df = df.copy()
        
        # 添加预测结果
        backtest_df['pred_class'] = pred_class
        
        # 获取每个类别的概率
        for i in range(pred_proba.shape[1]):
            backtest_df[f'prob_class_{i}'] = pred_proba[:, i]
        
        # 定义信号
        # 这里假设类别0: 震荡, 类别1: 上涨, 类别2: 下跌 (根据实际标签映射可能需要调整)
        # 调整信号逻辑以匹配实际的标签编码
        if len(self.trade_history) > 0:
            # 如果有交易历史，使用实际的标签映射
            label_mapping = {0: 0, 1: 1, 2: -1}  # 示例映射
        else:
            # 默认映射：类别0: 震荡(0), 类别1: 大涨(1), 类别2: 大跌(-1)
            label_mapping = {0: 0, 1: 1, 2: -1}
        
        # 计算信号强度（使用预测概率作为置信度）
        backtest_df['signal_strength'] = 0.0
        
        # 为每个类别计算信号强度
        for i, action in label_mapping.items():
            if i < pred_proba.shape[1]:  # 确保类别索引有效
                if action == 1:  # 上涨信号
                    backtest_df.loc[backtest_df['pred_class'] == i, 'signal_strength'] = \
                        backtest_df.loc[backtest_df['pred_class'] == i, f'prob_class_{i}']
                elif action == -1:  # 下跌信号
                    backtest_df.loc[backtest_df['pred_class'] == i, 'signal_strength'] = \
                        -backtest_df.loc[backtest_df['pred_class'] == i, f'prob_class_{i}']
        
        # 根据阈值过滤信号
        backtest_df['signal'] = 0
        backtest_df.loc[backtest_df['signal_strength'] > signal_threshold, 'signal'] = 1
        backtest_df.loc[backtest_df['signal_strength'] < -signal_threshold, 'signal'] = -1
        
        # 初始化回测变量
        position = 0  # 当前仓位: 1=多头, -1=空头, 0=无
        capital = self.initial_capital
        holdings = 0
        entry_price = 0
        trade_count = 0
        winning_trades = 0
        trade_history = []
        max_capital = capital
        drawdown = 0
        drawdown_start = None
        
        # 计算每笔交易的资金分配（使用固定比例）
        position_size_pct = self.max_position_size
        
        # 回测主循环
        for idx, row in backtest_df.iterrows():
            current_price = row['close']
            signal = row['signal']
            timestamp = idx
            
            # 计算当前总资产价值
            if position == 1:  # 多头
                current_value = capital + (current_price - entry_price) * holdings
            elif position == -1:  # 空头
                current_value = capital + (entry_price - current_price) * holdings
            else:  # 无仓位
                current_value = capital
            
            # 更新最大资本值和回撤
            if current_value > max_capital:
                max_capital = current_value
                drawdown = 0
                drawdown_start = None
            else:
                current_drawdown = (max_capital - current_value) / max_capital * 100
                if current_drawdown > drawdown:
                    drawdown = current_drawdown
                    if drawdown_start is None:
                        drawdown_start = timestamp
            
            # 记录当前状态
            backtest_df.loc[idx, 'capital'] = capital
            backtest_df.loc[idx, 'total_value'] = current_value
            backtest_df.loc[idx, 'position'] = position
            backtest_df.loc[idx, 'max_capital'] = max_capital
            backtest_df.loc[idx, 'drawdown'] = drawdown
            
            # 检查止损止盈
            should_exit = False
            exit_reason = ''
            
            if position != 0:
                # 计算持仓盈亏百分比
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # 检查止损
                if self.stop_loss > 0 and pnl_pct < -self.stop_loss:
                    should_exit = True
                    exit_reason = 'stop_loss'
                # 检查止盈
                elif self.take_profit > 0 and pnl_pct > self.take_profit:
                    should_exit = True
                    exit_reason = 'take_profit'
            
            # 交易逻辑
            if (position == 0 and signal != 0) or should_exit or (position != 0 and signal != position):
                # 平仓现有头寸
                if position != 0:
                    # 计算平仓价格（考虑滑点）
                    if position == 1:  # 多头平仓
                        exit_price = current_price - self.slippage
                    else:  # 空头平仓
                        exit_price = current_price + self.slippage
                    
                    # 计算盈亏
                    if position == 1:
                        pnl = (exit_price - entry_price) * holdings
                    else:
                        pnl = (entry_price - exit_price) * holdings
                    
                    # 扣除交易成本
                    cost = abs(holdings) * exit_price * self.transaction_cost
                    pnl_net = pnl - cost
                    
                    # 更新资本
                    capital += pnl_net
                    
                    # 判断是否盈利
                    is_winning = pnl_net > 0
                    if is_winning:
                        winning_trades += 1
                    
                    # 记录交易
                    trade_record = {
                        'trade_id': trade_count,
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'size': holdings,
                        'pnl': pnl,
                        'pnl_net': pnl_net,
                        'pnl_pct': pnl_pct,
                        'holding_period': (timestamp - entry_time).total_seconds() / 60,  # 分钟
                        'cost': cost,
                        'is_winning': is_winning,
                        'exit_reason': exit_reason
                    }
                    trade_history.append(trade_record)
                    trade_count += 1
                    
                    # 重置持仓
                    position = 0
                    holdings = 0
                
                # 开新仓
                if signal != 0 and not should_exit:
                    # 计算持仓大小
                    position_value = capital * position_size_pct
                    holdings = int(position_value / current_price)
                    
                    # 计算开仓价格（考虑滑点）
                    if signal == 1:  # 多头
                        entry_price = current_price + self.slippage
                    else:  # 空头
                        entry_price = current_price - self.slippage
                    
                    # 更新持仓
                    position = signal
                    entry_time = timestamp
        
        # 记录交易历史
        self.trade_history = pd.DataFrame(trade_history)
        self.backtest_results = backtest_df
        
        backtest_time = time.time() - start_time
        self.logger.info(f"回测完成，耗时: {backtest_time:.2f} 秒")
        self.logger.info(f"总交易次数: {trade_count}")
        
        return backtest_df
    
    def calculate_performance_metrics(self, backtest_df: pd.DataFrame) -> Dict:
        """
        计算性能指标
        
        Args:
            backtest_df: 回测结果DataFrame
            
        Returns:
            Dict: 性能指标
        """
        self.logger.info("计算性能指标")
        
        # 计算每日收益率
        if not isinstance(backtest_df.index, pd.DatetimeIndex):
            self.logger.warning("回测结果索引不是DatetimeIndex，尝试转换")
            try:
                backtest_df.index = pd.to_datetime(backtest_df.index)
            except:
                self.logger.error("无法转换索引为DatetimeIndex，使用原始索引计算")
        
        # 计算日收益率
        daily_returns = backtest_df['total_value'].resample('D').last().pct_change()
        
        # 计算基本指标
        final_value = backtest_df['total_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # 计算年化收益率
        trading_days = len(daily_returns.dropna())
        if trading_days > 0:
            annual_return = (pow(final_value / self.initial_capital, 252 / trading_days) - 1) * 100
        else:
            annual_return = 0
        
        # 计算波动率
        if len(daily_returns.dropna()) > 1:
            annual_volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        else:
            annual_volatility = 0
            sharpe_ratio = 0
        
        # 计算最大回撤
        max_drawdown = backtest_df['drawdown'].max()
        
        # 计算卡尔玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 交易指标
        trade_metrics = {}
        if self.trade_history is not None and len(self.trade_history) > 0:
            total_trades = len(self.trade_history)
            winning_trades = len(self.trade_history[self.trade_history['is_winning'] == True])
            win_rate = (winning_trades / total_trades) * 100
            
            # 计算盈亏比
            avg_winning_pnl = self.trade_history[self.trade_history['is_winning'] == True]['pnl_net'].mean()
            avg_losing_pnl = self.trade_history[self.trade_history['is_winning'] == False]['pnl_net'].mean()
            profit_factor = abs(avg_winning_pnl / avg_losing_pnl) if avg_losing_pnl != 0 else 0
            
            # 平均持仓时间
            avg_holding_period = self.trade_history['holding_period'].mean()
            
            trade_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_winning_pnl': avg_winning_pnl,
                'avg_losing_pnl': avg_losing_pnl,
                'avg_holding_period_minutes': avg_holding_period,
                'max_win_pnl': self.trade_history['pnl_net'].max(),
                'max_loss_pnl': self.trade_history['pnl_net'].min()
            }
        
        # 组装所有指标
        performance_metrics = {
            'basic': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'annual_return_pct': annual_return,
                'annual_volatility_pct': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'trading_days': trading_days
            },
            'trade': trade_metrics
        }
        
        return performance_metrics
    
    def analyze_by_market_state(self, backtest_df: pd.DataFrame) -> Dict:
        """
        按市场状态分析模型表现
        
        Args:
            backtest_df: 回测结果DataFrame
            
        Returns:
            Dict: 不同市场状态下的表现
        """
        self.logger.info("按市场状态分析模型表现")
        
        # 复制数据
        analysis_df = backtest_df.copy()
        
        # 计算市场状态指标（例如波动率）
        if 'high' in analysis_df.columns and 'low' in analysis_df.columns and 'close' in analysis_df.columns:
            # 计算日内波动率
            analysis_df['daily_volatility'] = (analysis_df['high'] - analysis_df['low']) / analysis_df['close'].shift(1) * 100
            
            # 计算收益率
            analysis_df['return'] = analysis_df['close'].pct_change() * 100
            
            # 确定市场状态
            # 高波动市场
            high_vol_threshold = analysis_df['daily_volatility'].quantile(0.7)
            # 低波动市场
            low_vol_threshold = analysis_df['daily_volatility'].quantile(0.3)
            
            # 定义市场状态
            analysis_df['market_state'] = 'normal'
            analysis_df.loc[analysis_df['daily_volatility'] > high_vol_threshold, 'market_state'] = 'high_volatility'
            analysis_df.loc[analysis_df['daily_volatility'] < low_vol_threshold, 'market_state'] = 'low_volatility'
            
            # 按市场状态分组计算指标
            state_performance = {}
            for state in ['low_volatility', 'normal', 'high_volatility']:
                state_df = analysis_df[analysis_df['market_state'] == state]
                
                if len(state_df) > 0:
                    # 计算该状态下的交易次数和胜率
                    if self.trade_history is not None:
                        state_trades = []
                        for _, trade in self.trade_history.iterrows():
                            # 检查交易是否在该市场状态期间
                            if trade['entry_time'] in state_df.index and trade['exit_time'] in state_df.index:
                                state_trades.append(trade)
                        
                        if state_trades:
                            state_trades_df = pd.DataFrame(state_trades)
                            winning_count = len(state_trades_df[state_trades_df['is_winning'] == True])
                            win_rate = (winning_count / len(state_trades_df)) * 100 if len(state_trades_df) > 0 else 0
                        else:
                            win_rate = 0
                    else:
                        win_rate = 0
                    
                    # 计算收益率
                    state_returns = state_df['total_value'].pct_change().dropna()
                    total_return = (state_df['total_value'].iloc[-1] / state_df['total_value'].iloc[0] - 1) * 100 if len(state_df) > 1 else 0
                    
                    state_performance[state] = {
                        'days': len(state_df),
                        'total_return_pct': total_return,
                        'avg_daily_return_pct': state_returns.mean() if len(state_returns) > 0 else 0,
                        'win_rate': win_rate
                    }
            
            return state_performance
        else:
            self.logger.error("缺少必要的价格列进行市场状态分析")
            return {}
    
    def plot_backtest_results(self, backtest_df: pd.DataFrame, performance_metrics: Dict):
        """
        绘制回测结果图表
        
        Args:
            backtest_df: 回测结果DataFrame
            performance_metrics: 性能指标
        """
        # 创建保存目录
        os.makedirs(self.backtest_results_dir, exist_ok=True)
        
        # 1. 绘制资产价值曲线
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df.index, backtest_df['total_value'], label='总资产价值')
        plt.plot(backtest_df.index, backtest_df['capital'], label='可用资金')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='初始资金')
        plt.title('资产价值变化')
        plt.xlabel('时间')
        plt.ylabel('价值')
        plt.legend()
        plt.grid(True)
        
        asset_path = os.path.join(self.backtest_results_dir, 'asset_value.png')
        plt.savefig(asset_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"资产价值图表已保存: {asset_path}")
        
        # 2. 绘制回撤曲线
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df.index, backtest_df['drawdown'], label='回撤 %')
        plt.title('回撤分析')
        plt.xlabel('时间')
        plt.ylabel('回撤 (%)')
        plt.grid(True)
        
        drawdown_path = os.path.join(self.backtest_results_dir, 'drawdown.png')
        plt.savefig(drawdown_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"回撤图表已保存: {drawdown_path}")
        
        # 3. 绘制交易信号和价格
        plt.figure(figsize=(14, 10))
        
        # 价格图
        plt.subplot(3, 1, 1)
        plt.plot(backtest_df.index, backtest_df['close'], label='价格')
        
        # 标记买入信号（信号从0变为1）
        buy_signals = backtest_df[(backtest_df['signal'].shift(1) != 1) & (backtest_df['signal'] == 1)]
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='买入信号')
        
        # 标记卖出信号（信号从0变为-1）
        sell_signals = backtest_df[(backtest_df['signal'].shift(1) != -1) & (backtest_df['signal'] == -1)]
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='卖出信号')
        
        plt.title('价格和交易信号')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # 信号强度图
        plt.subplot(3, 1, 2)
        plt.plot(backtest_df.index, backtest_df['signal_strength'], label='信号强度')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('信号强度')
        plt.ylabel('强度')
        plt.grid(True)
        
        # 持仓图
        plt.subplot(3, 1, 3)
        plt.plot(backtest_df.index, backtest_df['position'], label='持仓')
        plt.title('持仓变化')
        plt.ylabel('持仓')
        plt.xlabel('时间')
        plt.grid(True)
        
        signal_path = os.path.join(self.backtest_results_dir, 'signals_and_positions.png')
        plt.tight_layout()
        plt.savefig(signal_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"信号和持仓图表已保存: {signal_path}")
        
        # 4. 绘制交易分布直方图
        if self.trade_history is not None and len(self.trade_history) > 0:
            plt.figure(figsize=(14, 6))
            
            # 盈亏分布
            plt.subplot(1, 2, 1)
            sns.histplot(self.trade_history['pnl_net'], bins=30, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('盈亏分布')
            plt.xlabel('净盈亏')
            plt.ylabel('频率')
            
            # 持仓时间分布
            plt.subplot(1, 2, 2)
            sns.histplot(self.trade_history['holding_period'], bins=30, kde=True)
            plt.title('持仓时间分布')
            plt.xlabel('持仓时间（分钟）')
            plt.ylabel('频率')
            
            trade_path = os.path.join(self.backtest_results_dir, 'trade_distributions.png')
            plt.tight_layout()
            plt.savefig(trade_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"交易分布图表已保存: {trade_path}")
        
        plt.close('all')
    
    def generate_performance_report(self, performance_metrics: Dict, market_state_analysis: Dict = None):
        """
        生成性能报告
        
        Args:
            performance_metrics: 性能指标
            market_state_analysis: 市场状态分析结果
        """
        timestamp = time.strftime('%y%m%d_%H%M')
        report_path = os.path.join(self.backtest_results_dir, f'performance_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("===== 交易策略性能报告 =====\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本指标
            f.write("\n--- 基本性能指标 ---")
            basic = performance_metrics['basic']
            f.write(f"\n初始资金: {basic['initial_capital']}")
            f.write(f"\n最终价值: {basic['final_value']:.2f}")
            f.write(f"\n总收益率: {basic['total_return_pct']:.2f}%")
            f.write(f"\n年化收益率: {basic['annual_return_pct']:.2f}%")
            f.write(f"\n年化波动率: {basic['annual_volatility_pct']:.2f}%")
            f.write(f"\n夏普比率: {basic['sharpe_ratio']:.2f}")
            f.write(f"\n最大回撤: {basic['max_drawdown_pct']:.2f}%")
            f.write(f"\n卡尔玛比率: {basic['calmar_ratio']:.2f}")
            f.write(f"\n交易天数: {basic['trading_days']}")
            
            # 交易指标
            f.write("\n\n--- 交易指标 ---")
            trade = performance_metrics['trade']
            if trade:
                f.write(f"\n总交易次数: {trade['total_trades']}")
                f.write(f"\n盈利交易: {trade['winning_trades']}")
                f.write(f"\n亏损交易: {trade['losing_trades']}")
                f.write(f"\n胜率: {trade['win_rate']:.2f}%")
                f.write(f"\n盈亏比: {trade['profit_factor']:.2f}")
                f.write(f"\n平均盈利: {trade['avg_winning_pnl']:.2f}")
                f.write(f"\n平均亏损: {trade['avg_losing_pnl']:.2f}")
                f.write(f"\n平均持仓时间: {trade['avg_holding_period_minutes']:.2f} 分钟")
                f.write(f"\n最大盈利: {trade['max_win_pnl']:.2f}")
                f.write(f"\n最大亏损: {trade['max_loss_pnl']:.2f}")
            else:
                f.write("\n暂无交易数据")
            
            # 市场状态分析
            if market_state_analysis:
                f.write("\n\n--- 市场状态分析 ---")
                for state, metrics in market_state_analysis.items():
                    f.write(f"\n\n{state.replace('_', ' ').title()}:")
                    f.write(f"\n  天数: {metrics['days']}")
                    f.write(f"\n  总收益率: {metrics['total_return_pct']:.2f}%")
                    f.write(f"\n  平均日收益率: {metrics['avg_daily_return_pct']:.4f}%")
                    f.write(f"\n  胜率: {metrics['win_rate']:.2f}%")
            
            # 配置信息
            f.write("\n\n--- 回测配置 ---")
            f.write(f"\n初始资金: {self.initial_capital}")
            f.write(f"\n交易成本: {self.transaction_cost * 100:.4f}%")
            f.write(f"\n滑点: {self.slippage}")
            f.write(f"\n止损: {self.stop_loss}%")
            f.write(f"\n止盈: {self.take_profit}%")
            f.write(f"\n最大仓位: {self.max_position_size * 100:.2f}%")
        
        self.logger.info(f"性能报告已保存: {report_path}")
        return report_path
    
    def sensitivity_analysis(self, backtest_df: pd.DataFrame, pred_class: np.ndarray, 
                            pred_proba: np.ndarray) -> pd.DataFrame:
        """
        进行参数敏感性分析
        
        Args:
            backtest_df: 回测结果DataFrame
            pred_class: 预测类别
            pred_proba: 预测概率
            
        Returns:
            pd.DataFrame: 敏感性分析结果
        """
        self.logger.info("进行参数敏感性分析")
        
        # 测试不同的信号阈值
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        sensitivity_results = []
        
        for threshold in thresholds:
            # 运行回测
            temp_backtest = self.run_backtest(backtest_df.copy(), pred_class, pred_proba, signal_threshold=threshold)
            
            # 计算性能指标
            metrics = self.calculate_performance_metrics(temp_backtest)
            
            # 记录结果
            result = {
                'signal_threshold': threshold,
                'total_return_pct': metrics['basic']['total_return_pct'],
                'annual_return_pct': metrics['basic']['annual_return_pct'],
                'sharpe_ratio': metrics['basic']['sharpe_ratio'],
                'max_drawdown_pct': metrics['basic']['max_drawdown_pct'],
                'total_trades': metrics['trade'].get('total_trades', 0),
                'win_rate': metrics['trade'].get('win_rate', 0),
                'profit_factor': metrics['trade'].get('profit_factor', 0)
            }
            sensitivity_results.append(result)
        
        # 创建结果DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # 保存结果
        timestamp = time.strftime('%y%m%d_%H%M')
        sensitivity_path = os.path.join(self.backtest_results_dir, f'sensitivity_analysis_{timestamp}.csv')
        sensitivity_df.to_csv(sensitivity_path, index=False)
        self.logger.info(f"敏感性分析结果已保存: {sensitivity_path}")
        
        # 绘制敏感性分析图表
        plt.figure(figsize=(14, 7))
        
        # 夏普比率 vs 信号阈值
        plt.subplot(2, 2, 1)
        plt.plot(sensitivity_df['signal_threshold'], sensitivity_df['sharpe_ratio'], marker='o')
        plt.title('夏普比率 vs 信号阈值')
        plt.xlabel('信号阈值')
        plt.ylabel('夏普比率')
        plt.grid(True)
        
        # 总收益率 vs 信号阈值
        plt.subplot(2, 2, 2)
        plt.plot(sensitivity_df['signal_threshold'], sensitivity_df['total_return_pct'], marker='o')
        plt.title('总收益率 vs 信号阈值')
        plt.xlabel('信号阈值')
        plt.ylabel('总收益率 (%)')
        plt.grid(True)
        
        # 交易次数 vs 信号阈值
        plt.subplot(2, 2, 3)
        plt.plot(sensitivity_df['signal_threshold'], sensitivity_df['total_trades'], marker='o')
        plt.title('交易次数 vs 信号阈值')
        plt.xlabel('信号阈值')
        plt.ylabel('交易次数')
        plt.grid(True)
        
        # 胜率 vs 信号阈值
        plt.subplot(2, 2, 4)
        plt.plot(sensitivity_df['signal_threshold'], sensitivity_df['win_rate'], marker='o')
        plt.title('胜率 vs 信号阈值')
        plt.xlabel('信号阈值')
        plt.ylabel('胜率 (%)')
        plt.grid(True)
        
        sensitivity_plot_path = os.path.join(self.backtest_results_dir, f'sensitivity_analysis_plot_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(sensitivity_plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"敏感性分析图表已保存: {sensitivity_plot_path}")
        
        plt.close('all')
        return sensitivity_df
    
    def save_backtest_results(self, backtest_df: pd.DataFrame, performance_metrics: Dict):
        """
        保存回测结果
        
        Args:
            backtest_df: 回测结果DataFrame
            performance_metrics: 性能指标
        """
        timestamp = time.strftime('%y%m%d_%H%M')
        
        # 保存回测数据
        backtest_path = os.path.join(self.backtest_results_dir, f'backtest_results_{timestamp}.csv')
        backtest_df.to_csv(backtest_path)
        self.logger.info(f"回测结果已保存: {backtest_path}")
        
        # 保存交易历史
        if self.trade_history is not None:
            trade_history_path = os.path.join(self.backtest_results_dir, f'trade_history_{timestamp}.csv')
            self.trade_history.to_csv(trade_history_path)
            self.logger.info(f"交易历史已保存: {trade_history_path}")
        
        # 保存性能指标
        import json
        metrics_path = os.path.join(self.backtest_results_dir, f'performance_metrics_{timestamp}.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            # 将numpy类型转换为Python原生类型以便JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # 递归转换所有numpy类型
            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert_numpy(d)
            
            serializable_metrics = recursive_convert(performance_metrics)
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"性能指标已保存: {metrics_path}")
    
    def get_trading_recommendations(self, performance_metrics: Dict) -> List[Dict]:
        """
        生成交易策略建议
        
        Args:
            performance_metrics: 性能指标
            
        Returns:
            List[Dict]: 交易建议列表
        """
        recommendations = []
        basic = performance_metrics['basic']
        trade = performance_metrics['trade']
        
        # 基于夏普比率的建议
        if basic['sharpe_ratio'] < 0.5:
            recommendations.append({
                'type': '风险警告',
                'message': '夏普比率低于0.5，策略风险较高，建议谨慎使用'
            })
        elif 0.5 <= basic['sharpe_ratio'] < 1.0:
            recommendations.append({
                'type': '风险提示',
                'message': '夏普比率在0.5-1.0之间，策略有一定风险，建议控制仓位'
            })
        else:
            recommendations.append({
                'type': '策略肯定',
                'message': '夏普比率大于1.0，策略风险调整后收益良好'
            })
        
        # 基于最大回撤的建议
        if basic['max_drawdown_pct'] > 30:
            recommendations.append({
                'type': '风险警告',
                'message': f'最大回撤达到{basic["max_drawdown_pct"]:.1f}%，建议设置更严格的止损'
            })
        
        # 基于胜率的建议
        if trade and 'win_rate' in trade:
            if trade['win_rate'] < 40:
                recommendations.append({
                    'type': '优化建议',
                    'message': '胜率低于40%，考虑提高信号阈值或优化特征'
                })
            elif 40 <= trade['win_rate'] < 60:
                recommendations.append({
                    'type': '策略评估',
                    'message': '胜率适中，关注盈亏比的提升'
                })
            else:
                recommendations.append({
                    'type': '策略肯定',
                    'message': '胜率较高，保持当前信号质量'
                })
        
        # 基于交易频率的建议
        if trade and 'total_trades' in trade and 'avg_holding_period_minutes' in trade:
            if trade['total_trades'] > 1000 and trade['avg_holding_period_minutes'] < 10:
                recommendations.append({
                    'type': '成本警告',
                    'message': '交易频率过高，可能导致交易成本侵蚀利润'
                })
        
        # 综合建议
        recommendations.append({
            'type': '综合建议',
            'message': f'基于当前表现，建议采用{min(0.1, basic["sharpe_ratio"]/20):.1%}的资金配置，并严格执行止损策略'
        })
        
        return recommendations


if __name__ == "__main__":
    # 测试回测模块
    print("交易回测模块已创建")
    print("请使用主程序调用此模块进行完整的模型训练和回测")