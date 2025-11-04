import numpy as np
import pandas as pd
import torch
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, time
import math

# 确保中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建目录
os.makedirs('./backtest_results', exist_ok=True)

class LSTMTradingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMTradingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入Batch Normalization
        self.input_bn = torch.nn.BatchNorm1d(input_dim)
        
        # LSTM层
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch Normalization层
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 输入Batch Normalization
        x = x.permute(0, 2, 1).contiguous()  # (batch, input_dim, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, seq_len, input_dim)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 第一个全连接层 + BN + Dropout
        out = self.fc1(out)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # 第二个全连接层 + BN + Dropout
        out = self.fc2(out)
        
        return out

# 定义EnhancedLSTMTradingModel类，用于加载保存的模型
class EnhancedLSTMTradingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(EnhancedLSTMTradingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入Batch Normalization
        self.input_bn = torch.nn.BatchNorm1d(input_dim)
        
        # LSTM层
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层 - 根据错误信息，中间层维度应该是64
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)
        
        # Dropout层
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch Normalization层
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(64)  # 中间层维度为64
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 输入Batch Normalization
        x = x.permute(0, 2, 1).contiguous()  # (batch, input_dim, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, seq_len, input_dim)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 第一个全连接层 + BN + Dropout
        out = self.fc1(out)
        out = self.bn2(out)  # 使用bn2，维度为64
        out = torch.relu(out)
        out = self.dropout(out)
        
        # 第二个全连接层
        out = self.fc2(out)
        
        return out

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
    """数据预处理 - 与训练时保持一致"""
    # 复制数据以避免修改原始数据
    df_processed = df.copy()
    
    # 计算基本价格特征
    df_processed['high_low_ratio'] = df_processed['high'] / df_processed['low']
    df_processed['open_close_diff'] = df_processed['open'] - df_processed['close']
    df_processed['price_range'] = df_processed['high'] - df_processed['low']
    
    # 计算收益率
    for i in [1, 3, 5, 10, 20]:
        df_processed[f'returns_{i}'] = df_processed['close'].pct_change(i)
    
    # 计算移动平均线
    for i in [5, 10, 20, 50]:
        df_processed[f'ma_{i}'] = df_processed['close'].rolling(window=i).mean()
        df_processed[f'ma_diff_{i}'] = df_processed['close'] - df_processed[f'ma_{i}']
        df_processed[f'ma_ratio_{i}'] = df_processed['close'] / df_processed[f'ma_{i}']
    
    # 计算相对强弱指数(RSI)
    delta = df_processed['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_processed['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_processed['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_processed['close'].ewm(span=26, adjust=False).mean()
    df_processed['macd'] = exp1 - exp2
    df_processed['signal_line'] = df_processed['macd'].ewm(span=9, adjust=False).mean()
    df_processed['macd_diff'] = df_processed['macd'] - df_processed['signal_line']
    
    # 交易量特征
    df_processed['volume_pct_change'] = df_processed['volume'].pct_change()
    df_processed['volume_ma_5'] = df_processed['volume'].rolling(window=5).mean()
    df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_ma_5']
    
    # 持仓量特征
    df_processed['position_pct_change'] = df_processed['position'].pct_change()
    df_processed['position_ma_5'] = df_processed['position'].rolling(window=5).mean()
    df_processed['position_ratio'] = df_processed['position'] / df_processed['position_ma_5']
    
    # 高低开收特征
    df_processed['oc_diff'] = df_processed['open'] - df_processed['close']
    df_processed['hl_ratio'] = df_processed['high'] / df_processed['low']
    df_processed['oc_ratio'] = df_processed['open'] / df_processed['close']
    
    # 创建目标变量 - 使用未来5分钟的价格变化作为预测目标
    df_processed['future_returns'] = df_processed['close'].pct_change(5).shift(-5)
    df_processed['target'] = 0
    df_processed.loc[df_processed['future_returns'] > 0.0005, 'target'] = 1  # 多单
    df_processed.loc[df_processed['future_returns'] < -0.0005, 'target'] = -1  # 空单
    
    # 填充缺失值 - 只对数值列应用中位数填充
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    return df_processed

def create_sequences(data, seq_length, feature_columns):
    """创建时间序列数据"""
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[feature_columns].iloc[i:i+seq_length].values)
    return np.array(X)

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    try:
        # 尝试直接加载完整模型文件
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print(f"成功加载完整模型")
        return model
    except Exception as e:
        print(f"直接加载模型失败: {e}")
        try:
            # 如果直接加载失败，尝试使用状态字典方式
            # 创建模型实例
            input_dim = 38
            hidden_dim = 128
            num_layers = 3
            output_dim = 3
            dropout = 0.2
            
            model = EnhancedLSTMTradingModel(input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
            
            # 加载状态字典
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"成功加载状态字典模型")
            return model
        except Exception as inner_e:
            print(f"加载状态字典也失败: {inner_e}")
            raise

def load_scaler(scaler_path):
    """加载特征缩放器"""
    print(f"加载缩放器: {scaler_path}")
    try:
        # 尝试使用joblib加载（通常训练脚本使用joblib）
        scaler = joblib.load(scaler_path)
    except:
        try:
            # 尝试使用pickle加载
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except:
            # 如果都失败，创建一个新的缩放器
            print("无法加载缩放器，创建新的缩放器")
            scaler = StandardScaler()
    return scaler

def is_restricted_time(dt):
    """检查是否为限制开仓时间"""
    t = dt.time()
    # 9:00-9:15
    if time(9, 0) <= t <= time(9, 15):
        return True
    # 14:45-15:00
    if time(14, 45) <= t <= time(15, 0):
        return True
    # 21:00-21:15
    if time(21, 0) <= t <= time(21, 15):
        return True
    # 22:45-23:00
    if time(22, 45) <= t <= time(23, 0):
        return True
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
    
    # 执行回测
    for i in range(len(backtest_df)):
        dt = backtest_df.iloc[i]['bob']
        pred = backtest_df.iloc[i]['prediction']
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
            continue
        
        # 如果有持仓
        if current_position != 0:
            # 动态计算止盈价
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
            
            # 多仓动态止盈条件
            if current_position == 1:
                # 计算动态止盈价：(开仓价+3) - min(连续预测为0的K线数, 2)，最低为开仓价+1
                adjustment = min(consecutive_zero_count, 2)  # 最多调整2
                dynamic_take_profit = last_entry_price + 3 - adjustment
                dynamic_take_profit = max(dynamic_take_profit, last_entry_price + 1)  # 最低为开仓价+1
                
                if high_price > dynamic_take_profit:
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
                    continue
            
            # 空单止盈条件（保持原有逻辑）
            elif current_position == -1 and low_price < last_entry_price - 3:
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
                continue
            
            # 检查预测是否为0的4根K线自动平仓逻辑
            if pred == 0:
                # 记录第一次预测为0的位置
                if zero_prediction_start_index == -1:
                    zero_prediction_start_index = i
                # 检查是否达到4根k线
                elif i - zero_prediction_start_index >= 3:  # 0,1,2,3共4根k线
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
                    continue
            else:
                # 预测不为0，重置计数
                zero_prediction_start_index = -1
                consecutive_zero_count = 0
            
            # 保持当前持仓
            backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = current_position
        
        # 如果为空仓，检查是否可以开仓
        else:
            # 检查是否为限制开仓时间
            if not is_restricted_time(dt):
                if pred == 1 or pred == -1:
                    # 开仓
                    backtest_df.iloc[i, backtest_df.columns.get_loc('signal')] = pred
                    backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = pred
                    last_entry_price = close_price
                    current_position = pred
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
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
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

def main():
    # 配置参数
    config = {
        'sequence_length': 20,
        'hidden_dim': 64,  # 从128改为64以匹配训练时的实际配置
        'num_layers': 3,
        'output_dim': 3,  # -1, 0, 1
        'dropout': 0.2,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'initial_capital': 1000000
    }
    
    print(f"使用设备: {config['device']}")
    
    # 数据文件路径
    data_file = 'C:/python_workspace/future_data/fu2510_1M.csv'
    
    # 模型和缩放器路径 - 使用完整模型文件
    model_path = 'C:/python_workspace/Processor/models_multi_file_optimized_3layers/3_layers_128hd_0.2do_optimized_full_model_251023_1348.pkl'
    scaler_path = './models_multi_file_optimized_3layers/multi_file_scaler.pkl'
    
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
    
    # 加载模型和缩放器
    model = load_model(model_path, config['device'])
    scaler = load_scaler(scaler_path)
    
    # 加载和预处理数据
    df = load_data(data_file)
    df_processed = preprocess_data(df, config)
    
    # 创建序列
    X = create_sequences(df_processed, config['sequence_length'], feature_columns)
    print(f"创建序列: {X.shape}")
    
    # 对特征进行缩放
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(-1, config['sequence_length'], X.shape[2])
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(config['device'])
    
    # 模型预测
    predictions = []
    raw_outputs = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            raw_outputs.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # 分析预测分布
    pred_array = np.array(predictions)
    print("预测索引分布:")
    print(np.bincount(pred_array))
    
    # 检查原始输出
    raw_outputs_array = np.array(raw_outputs)
    print(f"原始输出形状: {raw_outputs_array.shape}")
    print(f"部分原始输出示例: {raw_outputs_array[:5]}")
    
    # 转换预测结果
    predicted_labels = np.array([-1, 0, 1])[predictions]
    print(f"预测标签分布:")
    print(np.unique(predicted_labels, return_counts=True))
    print(f"预测完成，预测标签形状: {predicted_labels.shape}")
    
    # 回测策略
    print("开始回测...")
    backtest_df = backtest_strategy(df, predicted_labels, config)
    
    # 计算性能指标
    metrics, cumulative_returns, drawdown = calculate_performance_metrics(backtest_df, config['initial_capital'])
    
    # 打印性能指标
    print("\n===== 回测性能指标 =====")
    print(f"年化收益率: {metrics['年化收益率']*100:.2f}%")
    print(f"年化波动率: {metrics['年化波动率']*100:.2f}%")
    print(f"夏普比率: {metrics['夏普比率']:.4f}")
    print(f"最大回撤: {metrics['最大回撤']*100:.2f}%")
    print(f"卡尔马比率: {metrics['卡尔马比率']:.4f}")
    print(f"总收益率: {metrics['总收益率']*100:.2f}%")
    print(f"总交易次数: {metrics['总交易次数']}")
    print(f"盈利交易次数: {metrics['盈利交易次数']}")
    print(f"亏损交易次数: {metrics['亏损交易次数']}")
    print(f"胜率: {metrics['胜率']*100:.2f}%")
    
    # 绘制性能图表
    plot_performance(backtest_df, cumulative_returns, drawdown, './backtest_results')
    print("\n回测性能图表已保存至: ./backtest_results/backtest_performance.png")
    
    # 确保backtest_results目录存在
    if not os.path.exists('./backtest_results'):
        os.makedirs('./backtest_results')
    
    # 使用时间戳创建唯一文件名，避免权限冲突
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'./backtest_results/backtest_results_{timestamp}.csv'
    pkl_file = f'./backtest_results/performance_metrics_{timestamp}.pkl'
    
    # 保存回测结果
    try:
        backtest_df.to_csv(csv_file, index=False)
        with open(pkl_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"回测结果已保存至: {csv_file}")
        print(f"性能指标已保存至: {pkl_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")
        # 尝试使用不同的方法保存
        try:
            # 删除旧文件（如果存在）
            if os.path.exists('./backtest_results/backtest_results.csv'):
                os.remove('./backtest_results/backtest_results.csv')
            if os.path.exists('./backtest_results/performance_metrics.pkl'):
                os.remove('./backtest_results/performance_metrics.pkl')
            # 重新保存
            backtest_df.to_csv('./backtest_results/backtest_results.csv', index=False)
            with open('./backtest_results/performance_metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f)
            print("回测结果已保存至: ./backtest_results/")
        except Exception as e2:
            print(f"再次保存失败: {e2}")
    
    # 以表格形式输出核心指标
    print("\n===== 核心性能指标表格 =====")
    print(f"{'指标':<12} {'值':<15} {'说明':<30}")
    print(f"{'-'*60}")
    print(f"{'年化收益率':<12} {metrics['年化收益率']*100:.2f}%  {'投资组合年度收益率':<30}")
    print(f"{'年化波动率':<12} {metrics['年化波动率']*100:.2f}%  {'反映收益率的波动性':<30}")
    print(f"{'夏普比率':<12} {metrics['夏普比率']:.4f}    {'每单位风险的超额收益':<30}")
    print(f"{'最大回撤':<12} {metrics['最大回撤']*100:.2f}%  {'最大亏损百分比':<30}")
    print(f"{'卡尔马比率':<12} {metrics['卡尔马比率']:.4f}    {'年化收益与最大回撤的比值':<30}")

if __name__ == "__main__":
    main()