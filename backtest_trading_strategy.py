import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, time
import math
import lightgbm as lgb

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建目录
os.makedirs('./backtest_results', exist_ok=True)

# 仅使用LightGBM模型进行回测，不再支持LSTM模型

def load_data(file_path):
    """加载期货数据"""
    print(f"加载数据: {file_path}")
    df = pd.read_csv(file_path)
    # 转换bob列为datetime类型
    df['bob'] = pd.to_datetime(df['bob'])
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['bob'].min()} 到 {df['bob'].max()}")
    return df

def preprocess_data(df, config):
    """数据预处理 - 生成与LightGBM模型训练时相同的50个特征"""
    # 复制数据以避免修改原始数据
    df_processed = df.copy()
    
    # 确保时间列为datetime类型
    if 'bob' in df_processed.columns:
        df_processed['bob'] = pd.to_datetime(df_processed['bob'])
    elif 'datetime' in df_processed.columns:
        df_processed['bob'] = pd.to_datetime(df_processed['datetime'])
    
    # 时间特征
    if 'bob' in df_processed.columns:
        df_processed['hour'] = df_processed['bob'].dt.hour
        df_processed['minute'] = df_processed['bob'].dt.minute
        df_processed['day_of_month'] = df_processed['bob'].dt.day
        # 交易时段特征
        df_processed['morning_session'] = ((df_processed['hour'] >= 9) & 
                                           (df_processed['hour'] < 11) | 
                                           (df_processed['hour'] == 11) & 
                                           (df_processed['minute'] <= 30)).astype(int)
        df_processed['afternoon_session'] = ((df_processed['hour'] >= 13) & 
                                             (df_processed['hour'] < 15)).astype(int)
    else:
        # 如果没有时间列，创建默认值
        df_processed['hour'] = 0
        df_processed['minute'] = 0
        df_processed['day_of_month'] = 1
        df_processed['morning_session'] = 0
        df_processed['afternoon_session'] = 0
    
    # 价格基本特征
    df_processed['price_range'] = df_processed['high'] - df_processed['low']
    df_processed['price_range_50'] = df_processed['price_range'].rolling(window=50).mean()
    df_processed['price_range_pct_10'] = df_processed['price_range'] / df_processed['close'].rolling(window=10).mean()
    df_processed['high_pct'] = (df_processed['high'] - df_processed['close'].shift(1)) / df_processed['close'].shift(1)
    df_processed['low_pct'] = (df_processed['low'] - df_processed['close'].shift(1)) / df_processed['close'].shift(1)
    
    # 收益率特征
    df_processed['return_1'] = df_processed['close'].pct_change(1)
    df_processed['return_10'] = df_processed['close'].pct_change(10)
    # 收益率滞后特征
    df_processed['return_lag_6'] = df_processed['close'].pct_change(6).shift(6)
    df_processed['return_lag_24'] = df_processed['close'].pct_change(24).shift(24)
    df_processed['return_lag_48'] = df_processed['close'].pct_change(48).shift(48)
    
    # 动量特征
    df_processed['momentum_5'] = df_processed['close'] - df_processed['close'].shift(5)
    df_processed['momentum_pct_20'] = (df_processed['close'] - df_processed['close'].shift(20)) / df_processed['close'].shift(20)
    
    # 波动率特征
    df_processed['volatility_20'] = df_processed['return_1'].rolling(window=20).std() * np.sqrt(20)
    df_processed['volatility_50'] = df_processed['return_1'].rolling(window=50).std() * np.sqrt(50)
    df_processed['realized_vol_10'] = df_processed['return_1'].rolling(window=10).std() * np.sqrt(10)
    
    # 统计特征
    df_processed['kurtosis_50'] = df_processed['return_1'].rolling(window=50).kurt()
    df_processed['skew_10'] = df_processed['return_1'].rolling(window=10).skew()
    df_processed['skew_50'] = df_processed['return_1'].rolling(window=50).skew()
    
    # 移动平均线特征
    df_processed['ma_5'] = df_processed['close'].rolling(window=5).mean()
    df_processed['ma_10'] = df_processed['close'].rolling(window=10).mean()
    df_processed['ma_20'] = df_processed['close'].rolling(window=20).mean()
    df_processed['ma_50'] = df_processed['close'].rolling(window=50).mean()  # 50周期简单移动平均线
    df_processed['ma_diff_5'] = df_processed['close'] - df_processed['ma_5']
    df_processed['ma_diff_10'] = df_processed['close'] - df_processed['ma_10']
    df_processed['ma_diff_pct_20'] = (df_processed['close'] - df_processed['ma_20']) / df_processed['ma_20']
    
    # ATR（Average True Range）指标计算
    df_processed['tr'] = np.maximum(df_processed['high'] - df_processed['low'], 
                                      np.abs(df_processed['high'] - df_processed['close'].shift(1)), 
                                      np.abs(df_processed['low'] - df_processed['close'].shift(1)))
    df_processed['atr_14'] = df_processed['tr'].rolling(window=14).mean()  # 14周期ATR
    
    # 移动平均线斜率
    df_processed['slope_10'] = df_processed['ma_10'].diff(10) / 10
    df_processed['slope_50'] = df_processed['ma_50'].diff(50) / 50 if 'ma_50' in df_processed.columns else 0
    df_processed['ma_slope_5'] = df_processed['ma_5'].diff()
    df_processed['ma_slope_10'] = df_processed['ma_10'].diff()
    
    # RSI特征
    # RSI 6
    delta_6 = df_processed['close'].diff()
    gain_6 = (delta_6.where(delta_6 > 0, 0)).rolling(window=6).mean()
    loss_6 = (-delta_6.where(delta_6 < 0, 0)).rolling(window=6).mean()
    rs_6 = gain_6 / loss_6
    df_processed['rsi_6'] = 100 - (100 / (1 + rs_6))
    # RSI 14
    delta_14 = df_processed['close'].diff()
    gain_14 = (delta_14.where(delta_14 > 0, 0)).rolling(window=14).mean()
    loss_14 = (-delta_14.where(delta_14 < 0, 0)).rolling(window=14).mean()
    rs_14 = gain_14 / loss_14
    df_processed['rsi_14'] = 100 - (100 / (1 + rs_14))
    # RSI移动平均
    df_processed['rsi_ma_14'] = df_processed['rsi_14'].rolling(window=14).mean()
    # RSI趋势
    df_processed['rsi_trend_6'] = df_processed['rsi_6'].diff(6)
    df_processed['rsi_trend_14'] = df_processed['rsi_14'].diff(14)
    df_processed['rsi_trend_21'] = df_processed['rsi_14'].diff(21)  # 使用rsi_14计算21周期趋势
    
    # MACD特征
    exp1 = df_processed['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_processed['close'].ewm(span=26, adjust=False).mean()
    df_processed['macd_line'] = exp1 - exp2
    df_processed['signal_line'] = df_processed['macd_line'].ewm(span=9, adjust=False).mean()
    df_processed['macd_hist'] = df_processed['macd_line'] - df_processed['signal_line']
    df_processed['macd_hist_ma'] = df_processed['macd_hist'].rolling(window=5).mean()
    df_processed['macd_hist_diff'] = df_processed['macd_hist'].diff()
    
    # 布林带特征
    rolling_mean = df_processed['close'].rolling(window=20).mean()
    rolling_std = df_processed['close'].rolling(window=20).std()
    df_processed['bb_upper_20'] = rolling_mean + (rolling_std * 2)
    df_processed['bb_lower_20'] = rolling_mean - (rolling_std * 2)
    df_processed['bb_upper_break_20'] = (df_processed['high'] > df_processed['bb_upper_20']).astype(int)
    df_processed['bb_lower_break_20'] = (df_processed['low'] < df_processed['bb_lower_20']).astype(int)
    
    # ATR特征
    high_low = df_processed['high'] - df_processed['low']
    high_close = np.abs(df_processed['high'] - df_processed['close'].shift(1))
    low_close = np.abs(df_processed['low'] - df_processed['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_processed['atr_5'] = true_range.rolling(window=5).mean()
    df_processed['atr_pct_10'] = df_processed['atr_5'] / df_processed['close'].rolling(window=10).mean()
    df_processed['atr_pct_30'] = df_processed['atr_5'] / df_processed['close'].rolling(window=30).mean()
    
    # 交易量特征
    df_processed['volume_ma_5'] = df_processed['volume'].rolling(window=5).mean()
    df_processed['volume_diff_pct_10'] = df_processed['volume'].pct_change(10)
    # 计算成交额
    df_processed['amount'] = df_processed['close'] * df_processed['volume']
    
    # 持仓量特征
    if 'position' in df_processed.columns:
        df_processed['position'] = df_processed['position']
    else:
        # 如果没有持仓量数据，使用默认值
        df_processed['position'] = df_processed['volume'].mean()
    
    # 填充缺失值 - 只对数值列应用中位数填充
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    return df_processed

# LightGBM不需要创建序列数据和加载PyTorch模型及缩放器

def is_restricted_time(dt):
    """
    检查是否为限制开仓时间 - 暂时禁用时间限制以验证模型效果
    """
    # 暂时禁用时间限制
    return False

def is_close_time(dt):
    """检查是否为强制平仓时间"""
    t = dt.time()
    # 14:58
    if t == time(14, 58):
        return True
    # 22:58
    if t == time(22, 58):
        return True
    return False

def backtest_strategy(df, predictions, config):
    """回测交易策略"""
    # 创建回测结果DataFrame
    backtest_df = df[config['sequence_length']-1:].copy()
    backtest_df['prediction'] = predictions
    backtest_df['position'] = 0  # 0: 空仓, 1: 多仓, -1: 空仓
    backtest_df['signal'] = 0    # 交易信号
    backtest_df['pnl'] = 0       # 盈亏
    backtest_df['cum_pnl'] = 0   # 累计盈亏
    
    # 初始化交易状态
    current_position = 0
    consecutive_zeros = 0
    last_entry_price = 0
    zero_prediction_start_index = -1  # 记录预测为0的起始位置
    consecutive_zero_count = 0  # 记录预测为0的连续K线数量
    position_enter_time = 0  # 记录持仓进入的索引位置
    trade_count = 0  # 记录交易次数
    
    # 策略参数 - 从配置文件中获取
    LOOKAHEAD = config.get('holding_period', 5)  # 持有期
    COOLDOWN_PERIOD = 5  # 冷却期 (暂时保留默认值)
    CONSECUTIVE_SIGNAL_THRESHOLD = config.get('consecutive_signals', 2)  # 连续信号要求
    STOP_LOSS_MULTIPLIER = config.get('stop_loss_multiplier', 3.0)  # 止损倍数 (基于ATR)
    TAKE_PROFIT_MULTIPLIER = config.get('take_profit_multiplier', 6.0)  # 止盈倍数 (基于ATR)
    
    # 添加调试信息
    print(f"=== 回测参数 ===")
    print(f"持有期: {LOOKAHEAD}")
    print(f"连续信号阈值: {CONSECUTIVE_SIGNAL_THRESHOLD}")
    print(f"止损倍数: {STOP_LOSS_MULTIPLIER}")
    print(f"止盈倍数: {TAKE_PROFIT_MULTIPLIER}")
    
    # 分析信号分布
    buy_signals = sum(predictions == 1)
    sell_signals = sum(predictions == -1)
    no_signals = sum(predictions == 0)
    print(f"=== 信号分布 ===")
    print(f"多头信号: {buy_signals}")
    print(f"空头信号: {sell_signals}")
    print(f"无信号: {no_signals}")
    print(f"总信号: {len(predictions)}")
    
    last_signals = []  # 存储最近的信号
    
    # 连续信号计数
    consecutive_bull_signals = 0
    consecutive_bear_signals = 0

    # 执行回测
    for i in range(len(backtest_df)):
        dt = backtest_df.iloc[i]['bob']
        pred = backtest_df.iloc[i]['prediction']
        # 更新连续信号计数
        if pred == 1:
            consecutive_bull_signals += 1
            consecutive_bear_signals = 0
        elif pred == -1:
            consecutive_bear_signals += 1
            consecutive_bull_signals = 0
        else:
            consecutive_bull_signals = 0
            consecutive_bear_signals = 0
        close_price = backtest_df.iloc[i]['close']
        high_price = backtest_df.iloc[i]['high']
        low_price = backtest_df.iloc[i]['low']
        
        # 检查是否需要强制平仓
        if is_close_time(dt):
            if current_position != 0:
                # 平仓
                backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                # 计算盈亏
                pnl = (close_price - last_entry_price) * current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                last_entry_price = 0
                current_position = 0
                consecutive_zeros = 0
                zero_prediction_start_index = -1
                consecutive_zero_count = 0
                position_enter_time = 0
            continue
        
        # 如果有持仓
        if current_position != 0:
            # 检查是否到达强制平仓时间（持有3个bar）
            force_close = False
            if i - position_enter_time >= LOOKAHEAD:
                force_close = True
            
            # 反向信号立即平仓
            if pred != 0 and pred != current_position:
                # 平仓
                backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                # 计算盈亏
                pnl = (close_price - last_entry_price) * current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                last_entry_price = 0
                current_position = 0
                consecutive_zeros = 0
                zero_prediction_start_index = -1
                consecutive_zero_count = 0
                position_enter_time = 0
                continue
            
            # 同向信号，延长持仓时间
            if pred != 0 and pred == current_position:
                # 重置持仓时间
                position_enter_time = i
            
            # 止损止盈机制
            current_atr = backtest_df.iloc[i].get('atr_14', 3)  # 使用atr_14代替atr_5
            
            # 增加日志输出
            print(f"  交易{trade_count}: 仓位={current_position}, ATR={current_atr:.4f}, 入场价={last_entry_price:.4f}, 当前价={close_price:.4f}")
            
            if current_position == 1:
                # 多头止损止盈
                stop_loss_multiplier = config.get('stop_loss_multiplier', 2.0)
                take_profit_multiplier = config.get('take_profit_multiplier', 4.0)
                stop_loss_price = last_entry_price - current_atr * stop_loss_multiplier
                take_profit_price = last_entry_price + current_atr * take_profit_multiplier
                
                print(f"  多头: 止损价={stop_loss_price:.4f}, 止盈价={take_profit_price:.4f}, 最高价={high_price:.4f}, 最低价={low_price:.4f}")
                
                # 检查止损
                if low_price <= stop_loss_price:
                    # 触发止损
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                    pnl = (stop_loss_price - last_entry_price) * current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                    print(f"  触发止损，盈利={pnl:.4f}")
                    last_entry_price = 0
                    current_position = 0
                    consecutive_zeros = 0
                    zero_prediction_start_index = -1
                    consecutive_zero_count = 0
                    position_enter_time = 0
                    continue
                
                # 检查止盈
                if high_price >= take_profit_price:
                    # 触发止盈
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                    pnl = (take_profit_price - last_entry_price) * current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                    print(f"  触发止盈，盈利={pnl:.4f}")
                    last_entry_price = 0
                    current_position = 0
                    consecutive_zeros = 0
                    zero_prediction_start_index = -1
                    consecutive_zero_count = 0
                    position_enter_time = 0
                    continue
            
            # 空单止损止盈
            elif current_position == -1:
                stop_loss_multiplier = config.get('stop_loss_multiplier', 2.0)
                take_profit_multiplier = config.get('take_profit_multiplier', 4.0)
                stop_loss_price = last_entry_price + current_atr * stop_loss_multiplier
                take_profit_price = last_entry_price - current_atr * take_profit_multiplier
                
                print(f"  空头: 止损价={stop_loss_price:.4f}, 止盈价={take_profit_price:.4f}, 最高价={high_price:.4f}, 最低价={low_price:.4f}")
                
                # 检查止损
                if high_price >= stop_loss_price:
                    # 触发止损
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                    pnl = (stop_loss_price - last_entry_price) * current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                    print(f"  触发止损，盈利={pnl:.4f}")
                    last_entry_price = 0
                    current_position = 0
                    consecutive_zeros = 0
                    zero_prediction_start_index = -1
                    consecutive_zero_count = 0
                    position_enter_time = 0
                    continue
                
                # 检查止盈
                if low_price <= take_profit_price:
                    # 触发止盈
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                    pnl = (take_profit_price - last_entry_price) * current_position
                    backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                    print(f"  触发止盈，盈利={pnl:.4f}")
                    last_entry_price = 0
                    current_position = 0
                    consecutive_zeros = 0
                    zero_prediction_start_index = -1
                    consecutive_zero_count = 0
                    position_enter_time = 0
                    continue
            
            # 优化：当连续预测为0时，结合ATR调整止损止盈
            if pred == 0:
                # 记录连续预测为0的K线数量
                if zero_prediction_start_index == -1:
                    zero_prediction_start_index = i
                    consecutive_zero_count = 1
                else:
                    consecutive_zero_count = i - zero_prediction_start_index + 1
            else:
                consecutive_zero_count = 0
                zero_prediction_start_index = -1
            
            # 持有时间达到强制平仓条件
            if force_close:
                # 平仓
                backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
                # 计算盈亏
                pnl = (close_price - last_entry_price) * current_position
                backtest_df.iloc[i, backtest_df.columns.get_loc('pnl')] = pnl
                last_entry_price = 0
                current_position = 0
                consecutive_zeros = 0
                zero_prediction_start_index = -1
                consecutive_zero_count = 0
                position_enter_time = 0
                continue
            
            # 保持当前持仓
            backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = current_position
        
        # 如果为空仓，检查是否可以开仓
        else:
            # 检查是否为限制开仓时间
            restricted = is_restricted_time(dt)
            
            if not restricted:
                # 直接基于模型预测的信号开仓
                if pred == 1:
                    # 多头信号：买入1手沪深300
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = 1
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 1
                    last_entry_price = close_price
                    current_position = 1
                    position_enter_time = i  # 记录持仓进入时间
                    trade_count += 1  # 增加交易次数
                    print(f"时间点 {dt} -> 成功开仓: 多头信号，价格 {close_price}")
                elif pred == -1:
                    # 空头信号：卖出1手沪深300
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = -1
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = -1
                    last_entry_price = close_price
                    current_position = -1
                    position_enter_time = i  # 记录持仓进入时间
                    trade_count += 1  # 增加交易次数
                    print(f"时间点 {dt} -> 成功开仓: 空头信号，价格 {close_price}")
            zero_prediction_start_index = -1

    # 计算累计盈亏
    backtest_df['cum_pnl'] = backtest_df['pnl'].cumsum()

    return backtest_df
def calculate_performance_metrics(backtest_df, initial_capital=1000000):
    """计算性能指标"""
    # 计算日收益率
    daily_returns = backtest_df.groupby(backtest_df['bob'].dt.date)['pnl'].sum() / initial_capital
    
    # 年化收益率
    total_return = backtest_df['cum_pnl'].iloc[-1] / initial_capital
    days = len(daily_returns) if len(daily_returns) > 0 else 1
    annual_return = (1 + total_return) ** (365 / days) - 1
    
    # 年化波动率
    daily_volatility = daily_returns.std() if len(daily_returns) > 1 else 0
    annual_volatility = daily_volatility * math.sqrt(365)
    
    # 夏普比率（无风险利率3%）
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if (annual_volatility > 0 and not np.isnan(annual_volatility)) else 0
    
    # 最大回撤
    cumulative_returns = 1 + backtest_df['cum_pnl'] / initial_capital
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 卡尔马比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # 交易统计
    total_trades = len(backtest_df[backtest_df['signal'] != 0])
    winning_trades = len(backtest_df[(backtest_df['pnl'] > 0) & (backtest_df['signal'] != 0)])
    losing_trades = len(backtest_df[(backtest_df['pnl'] < 0) & (backtest_df['signal'] != 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    metrics = {
        '年化收益率': annual_return,
        '年化波动率': annual_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡尔马比率': calmar_ratio,
        '总收益率': total_return,
        '总交易次数': total_trades,
        '盈利交易次数': winning_trades,
        '亏损交易次数': losing_trades,
        '胜率': win_rate
    }
    
    return metrics, cumulative_returns, drawdown

def plot_performance(backtest_df, cumulative_returns, drawdown, save_dir):
    """绘制回测性能图表"""
    # 绘制累计盈亏曲线
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(backtest_df['bob'], cumulative_returns)
    plt.title('累计收益率曲线')
    plt.grid(True)
    
    # 绘制回撤曲线
    plt.subplot(3, 1, 2)
    plt.plot(backtest_df['bob'], drawdown)
    plt.title('回撤曲线')
    plt.grid(True)
    
    # 绘制交易信号
    plt.subplot(3, 1, 3)
    plt.plot(backtest_df['bob'], backtest_df['close'], alpha=0.5, label='价格')
    buy_signals = backtest_df[backtest_df['signal'] == 1]
    sell_signals = backtest_df[backtest_df['signal'] == -1]
    plt.scatter(buy_signals['bob'], buy_signals['close'], color='red', marker='^', label='买入信号')
    plt.scatter(sell_signals['bob'], sell_signals['close'], color='green', marker='v', label='卖出信号')
    plt.title('交易信号')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'backtest_performance.png'), dpi=300)
    plt.close()

def apply_confidence_threshold(raw_outputs_array, threshold):
    """
    对模型输出应用置信度阈值过滤
    
    参数:
        outputs: 模型原始输出 (shape: [n_samples, 3])
        threshold: 置信度阈值
        
    返回:
        predictions: 过滤后的预测结果 (-1, 0, 1)
        max_probabilities: 每个样本的最大概率值
        probabilities: 所有类别的概率值
    """
    # 打印原始输出的分布情况
    print(f"原始输出形状: {raw_outputs_array.shape}")
    print(f"原始输出范围: {raw_outputs_array.min()} 到 {raw_outputs_array.max()}")
    print(f"原始输出均值: {raw_outputs_array.mean()}")
    print(f"原始输出前5行:\n{raw_outputs_array[:5]}")
    
    # 应用softmax转换为概率
    from scipy.special import softmax
    probabilities = softmax(raw_outputs_array, axis=1)
    
    # 获取最大概率和对应类别
    max_probabilities = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    
    # 将类别转换为-1, 0, 1
    filtered_predictions = np.array([-1, 0, 1])[predicted_classes]
    
    # 应用置信度阈值：概率低于阈值的信号设为0
    filtered_predictions[max_probabilities < threshold] = 0
    
    return filtered_predictions, max_probabilities, probabilities

def main():
    # 配置参数
    config = {
        'initial_capital': 1000000,
        'model_type': 'lightgbm',  # 使用LightGBM模型
         'sequence_length': 20,  # 即使使用LightGBM，backtest_strategy函数仍需要此参数
         'stop_loss_multiplier': 2.0,  # ATR倍数止损 (降低以减少止损触发)
    'take_profit_multiplier': 5.0,  # ATR倍数止盈 (提高以增加盈利潜力)
    'holding_period': 5,  # 持仓期限 (与模型训练窗口保持一致)
    'consecutive_signals': 1,  # 连续相同信号次数 (降低以增加交易机会)
    'volatility_filter': 0.01,  # 波动性过滤阈值 (降低以增加交易机会)
    'momentum_confirmation': False  # 关闭动量确认 (增加交易机会)
    }
    
    print(f"使用LightGBM模型进行回测")
    
    # 数据文件路径
    data_file = r'C:\LightGBM_Prediction_Singal\backtest_data\fu2510_1M.csv'
    
    # 使用指定的LightGBM模型文件（使用原始字符串避免转义问题）
    # 根据日志信息，模型文件是txt格式
    model_path = r'C:\LightGBM_Prediction_Singal\models_lgbm\lgbm_model_v1.0_251106_1643.txt'
    metadata_path = r'C:\LightGBM_Prediction_Singal\models_lgbm\lgbm_model_v1.0_251106_1643_metadata.pkl'
    
    # 特征列 - 与训练时保持一致（38个特征）
    feature_columns = [
        # 价格基本特征
        'high_low_ratio', 'open_close_diff', 'price_range',
        # 收益率特征
        'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
        # 移动平均线特征
        'ma_5', 'ma_diff_5', 'ma_ratio_5', 
        'ma_10', 'ma_diff_10', 'ma_ratio_10', 
        'ma_20', 'ma_diff_20', 'ma_ratio_20', 
        'ma_50', 'ma_diff_50', 'ma_ratio_50',
        # 技术指标
        'rsi', 'macd', 'signal_line', 'macd_diff',
        # 交易量特征
        'volume_pct_change', 'volume_ma_5', 'volume_ratio',
        # 持仓量特征
        'position_pct_change', 'position_ma_5', 'position_ratio',
        # 价格关系特征
        'oc_diff', 'hl_ratio', 'oc_ratio',
        # 基本价格和交易量特征（删除一个以确保38个特征）
        'close', 'high', 'low', 'open', 'volume'
    ]
    
    # 加载和预处理数据
    df = load_data(data_file)
    df_processed = preprocess_data(df, config)
    
    # 加载LightGBM模型
    print(f"加载模型: {model_path}")
    try:
        # 加载LightGBM模型
        model = lgb.Booster(model_file=model_path)
        print("LightGBM模型加载成功")
        
        # 加载元数据获取特征列信息
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 如果元数据是字典，尝试获取特征列
            if isinstance(metadata, dict) and 'feature_columns' in metadata:
                feature_columns = metadata['feature_columns']
                print(f"从元数据获取特征列，数量: {len(feature_columns)}")
            # 如果元数据是numpy数组，假设它包含特征列名称
            elif isinstance(metadata, np.ndarray):
                feature_columns = metadata.flatten().tolist()
                print(f"从元数据数组获取特征列，数量: {len(feature_columns)}")
            else:
                raise ValueError("元数据格式不符合预期")
        except Exception as e:
            print(f"无法从元数据获取特征列: {e}")
            # 使用默认特征集
            feature_columns = [
                'atr_5', 'return_1', 'bb_upper_20', 'minute', 'price_range_50',
                'volume_ma_5', 'atr_pct_30', 'atr_pct_10', 'position', 'ma_diff_5',
                'volatility_50', 'kurtosis_50', 'macd_hist_diff', 'realized_vol_10',
                'volume_diff_pct_10', 'amount', 'slope_50', 'volatility_20', 'rsi_ma_14',
                'day_of_month', 'rsi_6', 'skew_50', 'price_range_pct_10', 'rsi_trend_6',
                'rsi_trend_21', 'skew_10', 'low_pct', 'ma_diff_10', 'rsi_trend_14', 'rsi_14',
                'macd_line', 'macd_hist', 'momentum_pct_20', 'slope_10', 'high_pct',
                'ma_diff_pct_20', 'hour', 'macd_hist_ma', 'return_10', 'ma_slope_10',
                'return_lag_24', 'ma_slope_5', 'return_lag_6', 'return_lag_48',
                'price_range', 'momentum_5', 'morning_session', 'afternoon_session',
                'bb_upper_break_20', 'bb_lower_break_20'
            ]
            print(f"使用默认特征集，数量: {len(feature_columns)}")
    except Exception as e:
        print(f"模型加载出错: {e}")
        raise
    
    # 确保数据中包含所有需要的特征
    available_features = [col for col in feature_columns if col in df_processed.columns]
    missing_features = [col for col in feature_columns if col not in df_processed.columns]
    
    if missing_features:
        print(f"警告: 缺失以下特征: {missing_features[:10]}...")  # 只显示前10个缺失特征
        print(f"缺失特征总数: {len(missing_features)}")
        # 使用可用的特征进行预测
        feature_columns = available_features
    
    print(f"实际使用的特征数量: {len(feature_columns)}")
    
    # 准备预测数据
    X_pred = df_processed[feature_columns].values
    print(f"预测数据形状: {X_pred.shape}")
    
    # 模型预测：获取原始logits（raw_score=True）
    logits = model.predict(X_pred, raw_score=True)
    
    # 添加调试信息：打印原始logits分布
    print(f"原始logits形状: {logits.shape}")
    print(f"部分原始logits示例: {logits[:5]}")
    
    # 转换logits为概率
    from scipy.special import softmax
    probabilities = softmax(logits, axis=1)
    
    # 计算最大概率和预测类别
    max_probs = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    
    # 添加调试信息：打印预测类别分布
    unique_classes, class_counts = np.unique(predicted_classes, return_counts=True)
    print(f"原始预测类别分布: {dict(zip(unique_classes, class_counts))}")
    print(f"最大概率统计: 均值={np.mean(max_probs):.4f}, 中位数={np.median(max_probs):.4f}, 最小值={np.min(max_probs):.4f}, 最大值={np.max(max_probs):.4f}")
    
    # LightGBM模型预测的类别通常是0,1,2，我们需要映射到-1,0,1
    # 假设类别0对应下跌(-1)，类别1对应震荡(0)，类别2对应上涨(1)
    class_mapping = {0: -1, 1: 0, 2: 1}
    predicted_classes_mapped = [class_mapping.get(cls, 0) for cls in predicted_classes]
    raw_outputs_array = logits  # 使用原始logits作为apply_confidence_threshold的输入
    print(f"原始输出形状: {raw_outputs_array.shape}")
    print(f"部分原始输出示例: {raw_outputs_array[:5]}")
    max_probs = np.max(probabilities, axis=1)
    print(f"原始输出的最大置信度分布:")
    print(f"平均值: {np.mean(max_probs)}")
    print(f"中位数: {np.median(max_probs)}")
    print(f"最小值: {np.min(max_probs)}")
    print(f"最大值: {np.max(max_probs)}")
    
    # 使用不同置信度阈值进行回测 - 提高阈值以过滤低置信度信号
    confidence_thresholds = [0.9, 0.95, 0.97, 0.98, 0.99]  # 调整为接近1.0的阈值以显示效果
    all_results = {}
    
    for threshold in confidence_thresholds:
        print(f"\n===== 使用置信度阈值 {threshold} 进行回测 =====")
        
        # 应用置信度阈值过滤预测结果
        filtered_predictions, max_probs, probabilities = apply_confidence_threshold(raw_outputs_array, threshold)
        
        # 截断预测结果以匹配回测数据长度
        seq_offset = config['sequence_length'] - 1
        filtered_predictions_truncated = filtered_predictions[seq_offset:]
        max_probs_truncated = max_probs[seq_offset:]
        probabilities_truncated = probabilities[seq_offset:]
        
        # 分析过滤后的预测分布
        print(f"过滤后的预测标签分布:")
        unique, counts = np.unique(filtered_predictions_truncated, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # 计算有效信号比例
        valid_signals = np.sum(filtered_predictions_truncated != 0)
        total_predictions = len(filtered_predictions_truncated)
        signal_ratio = (valid_signals / total_predictions * 100) if total_predictions > 0 else 0
        print(f"有效信号比例: {signal_ratio:.2f}% ({valid_signals}/{total_predictions})\n")
        
        # 回测策略
        print("开始回测...")
        backtest_df = backtest_strategy(df_processed, filtered_predictions_truncated, config)
        
        # 添加概率信息到回测结果中，确保包含所有bar数据
        result_df = df[config['sequence_length']-1:].copy()
        result_df = pd.concat([result_df.reset_index(drop=True), 
                              backtest_df[['prediction', 'position', 'signal', 'pnl', 'cum_pnl']].reset_index(drop=True)], 
                             axis=1)
        
        # 添加概率信息 - 使用截断后的数据以匹配回测结果长度
        result_df['confidence'] = max_probs_truncated
        result_df['prob_neg1'] = probabilities_truncated[:, 0]
        result_df['prob_0'] = probabilities_truncated[:, 1]
        result_df['prob_1'] = probabilities_truncated[:, 2]
        
        # 计算性能指标
        metrics, cumulative_returns, drawdown = calculate_performance_metrics(result_df, config['initial_capital'])
        
        # 打印性能指标
        print(f"\n----- 置信度阈值 {threshold} 的性能指标 -----")
        print(f"年化收益率: {metrics['年化收益率']*100:.2f}%")
        print(f"年化波动率: {metrics['年化波动率']*100:.2f}%")
        print(f"夏普比率: {metrics['夏普比率']:.4f}")
        print(f"最大回撤: {metrics['最大回撤']*100:.2f}%")
        print(f"总收益率: {metrics['总收益率']*100:.2f}%")
        print(f"总交易次数: {metrics['总交易次数']}")
        print(f"胜率: {metrics['胜率']*100:.2f}%")
        
        # 保存结果
        all_results[threshold] = {
            'df': result_df,
            'metrics': metrics,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown
        }
    
    # 确保backtest_results目录存在
    if not os.path.exists('./backtest_results'):
        os.makedirs('./backtest_results')
    
    # 使用时间戳创建唯一文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存每个置信度阈值的结果
    for threshold in confidence_thresholds:
        result_df = all_results[threshold]['df']
        metrics = all_results[threshold]['metrics']
        
        # 创建包含置信度的文件名
        threshold_str = str(threshold).replace('.', '')
        csv_file = f'./backtest_results/backtest_results_{timestamp}_conf{threshold_str}.csv'
        pkl_file = f'./backtest_results/performance_metrics_{timestamp}_conf{threshold_str}.pkl'
        
        # 保存回测结果（包含所有bar数据和概率信息）
        try:
            result_df.to_csv(csv_file, index=False)
            with open(pkl_file, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"\n置信度 {threshold} 的回测结果已保存至: {csv_file}")
            print(f"性能指标已保存至: {pkl_file}")
        except Exception as e:
            print(f"保存置信度 {threshold} 结果时出错: {e}")
    
    # 绘制所有置信度阈值的性能对比图
    plt.figure(figsize=(15, 10))
    
    # 累计收益率对比
    plt.subplot(2, 1, 1)
    for threshold in confidence_thresholds:
        result_df = all_results[threshold]['df']
        plt.plot(result_df['bob'], result_df['cum_pnl'], label=f'置信度 {threshold}')
    plt.title('不同置信度阈值的累计盈亏曲线对比')
    plt.legend()
    plt.grid(True)
    
    # 回撤对比
    plt.subplot(2, 1, 2)
    for threshold in confidence_thresholds:
        result_df = all_results[threshold]['df']
        cumulative_returns = 1 + result_df['cum_pnl'] / config['initial_capital']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        plt.plot(result_df['bob'], drawdown, label=f'置信度 {threshold}')
    plt.title('不同置信度阈值的回撤曲线对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    comparison_file = f'./backtest_results/confidence_comparison_{timestamp}.png'
    plt.savefig(comparison_file, dpi=300)
    plt.close()
    print(f"\n置信度阈值对比图表已保存至: {comparison_file}")
    
    # 以表格形式输出所有置信度阈值的核心指标对比
    print("\n===== 不同置信度阈值的核心性能指标对比 =====")
    print(f"{'置信度':<10} {'年化收益率':<15} {'夏普比率':<15} {'最大回撤':<15} {'总交易次数':<15} {'胜率':<10}")
    print(f"{'-'*85}")
    
    for threshold in confidence_thresholds:
        metrics = all_results[threshold]['metrics']
        print(f"{threshold:<10} {metrics['年化收益率']*100:.2f}%{'':<10} {metrics['夏普比率']:.4f}{'':<10} {metrics['最大回撤']*100:.2f}%{'':<10} {metrics['总交易次数']:<15} {metrics['胜率']*100:.2f}%")


if __name__ == "__main__":
    main()