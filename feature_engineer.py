#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 特征工程模块

此模块负责生成各种技术指标和特征
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 默认导入talib库
import talib
logging.info("talib库导入成功，使用talib计算所有技术指标")

from config import Config

# 设置日志
logger = logging.getLogger('feature_engineer')

# 确保所有处理器使用UTF-8编码
for handler in logging.root.handlers:
    if hasattr(handler, 'set_encoding'):
        handler.set_encoding('utf-8')
    # 移除尝试设置只读的stream.encoding属性


class FeatureEngineer:
    """
    特征工程类，负责生成各种技术指标和特征
    """
    
    def __init__(self):
        """初始化特征工程器"""
        self.logger = logger
        self.logger.info("特征工程器初始化完成")
        self.feature_names = []
    
    def calculate_basic_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础价格特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了基础价格特征的数据
        """
        df_features = df.copy()
        
        # OHLC变换
        df_features['price_range'] = df_features['high'] - df_features['low']
        df_features['price_change'] = df_features['close'] - df_features['open']
        df_features['price_change_pct'] = df_features['price_change'] / df_features['open']
        df_features['high_pct'] = (df_features['high'] - df_features['open']) / df_features['open']
        df_features['low_pct'] = (df_features['open'] - df_features['low']) / df_features['open']
        
        # 收益率计算
        df_features['return_1'] = df_features['close'].pct_change(1)
        df_features['return_5'] = df_features['close'].pct_change(5)
        df_features['return_10'] = df_features['close'].pct_change(10)
        
        # 波动率特征
        df_features['realized_vol_5'] = df_features['return_1'].rolling(5).std() * np.sqrt(5)
        df_features['realized_vol_10'] = df_features['return_1'].rolling(10).std() * np.sqrt(10)
        
        # 添加到特征列表
        new_features = ['price_range', 'price_change', 'price_change_pct', 'high_pct', 'low_pct',
                        'return_1', 'return_5', 'return_10', 'realized_vol_5', 'realized_vol_10']
        self.feature_names.extend(new_features)
        
        return df_features
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算移动平均线特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了移动平均线特征的数据
        """
        df_features = df.copy()
        
        for window in Config.MA_WINDOWS:
            # 直接使用talib计算简单移动平均线和指数移动平均线
            df_features[f'ma_{window}'] = talib.SMA(df_features['close'].values, timeperiod=window)
            df_features[f'ema_{window}'] = talib.EMA(df_features['close'].values, timeperiod=window)
            
            # 计算价格与均线的偏差
            df_features[f'ma_diff_{window}'] = df_features['close'] - df_features[f'ma_{window}']
            df_features[f'ma_diff_pct_{window}'] = df_features[f'ma_diff_{window}'] / df_features[f'ma_{window}']
            # 计算均线斜率
            df_features[f'ma_slope_{window}'] = df_features[f'ma_{window}'].diff(3) / 3
        
        # 添加到特征列表
        for window in Config.MA_WINDOWS:
            new_features = [f'ma_{window}', f'ema_{window}', f'ma_diff_{window}', 
                            f'ma_diff_pct_{window}', f'ma_slope_{window}']
            self.feature_names.extend(new_features)
        
        return df_features
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算RSI指标
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了RSI特征的数据
        """
        df_features = df.copy()
        
        for window in Config.RSI_WINDOWS:
            # 直接使用talib计算RSI
            df_features[f'rsi_{window}'] = talib.RSI(df_features['close'].values, timeperiod=window)
            
            # 计算RSI的移动平均线和趋势
            df_features[f'rsi_ma_{window}'] = df_features[f'rsi_{window}'].rolling(window=window).mean()
            df_features[f'rsi_trend_{window}'] = df_features[f'rsi_{window}'].diff(window)
        
        # 添加到特征列表
        for window in Config.RSI_WINDOWS:
            new_features = [f'rsi_{window}', f'rsi_ma_{window}', f'rsi_trend_{window}']
            self.feature_names.extend(new_features)
        
        return df_features
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算MACD指标
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了MACD特征的数据
        """
        df_features = df.copy()
        
        # 直接使用talib计算MACD
        # talib.MACD返回 (macd_line, signal_line, histogram)
        macd_line, macd_signal, macd_hist = talib.MACD(
            df_features['close'].values,
            fastperiod=Config.MACD_FAST,
            slowperiod=Config.MACD_SLOW,
            signalperiod=Config.MACD_SIGNAL
        )
        df_features['macd_line'] = macd_line
        df_features['macd_signal'] = macd_signal
        df_features['macd_hist'] = macd_hist
        
        # MACD的特征扩展
        df_features['macd_hist_ma'] = df_features['macd_hist'].rolling(window=5).mean()
        df_features['macd_hist_diff'] = df_features['macd_hist'].diff()
        df_features['macd_crossover'] = 0
        df_features.loc[(df_features['macd_line'] > df_features['macd_signal']) & 
                        (df_features['macd_line'].shift(1) <= df_features['macd_signal'].shift(1)), 'macd_crossover'] = 1
        df_features.loc[(df_features['macd_line'] < df_features['macd_signal']) & 
                        (df_features['macd_line'].shift(1) >= df_features['macd_signal'].shift(1)), 'macd_crossover'] = -1
        
        # 添加到特征列表
        new_features = ['macd_line', 'macd_signal', 'macd_hist', 'macd_hist_ma', 
                        'macd_hist_diff', 'macd_crossover']
        self.feature_names.extend(new_features)
        
        return df_features
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算布林带指标
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了布林带特征的数据
        """
        df_features = df.copy()
        
        for window in Config.BB_WINDOWS:
            # 直接使用talib计算布林带
            # talib.BBANDS返回 (upperband, middleband, lowerband)
            upper_band, middle_band, lower_band = talib.BBANDS(
                df_features['close'].values,
                timeperiod=window,
                nbdevup=Config.BB_STD,
                nbdevdn=Config.BB_STD,
                matype=0  # 简单移动平均
            )
            df_features[f'bb_upper_{window}'] = upper_band
            df_features[f'bb_middle_{window}'] = middle_band
            df_features[f'bb_lower_{window}'] = lower_band
            
            # 带宽指标
            df_features[f'bb_bandwidth_{window}'] = (upper_band - lower_band) / middle_band
            
            # 价格相对于布林带的位置
            df_features[f'bb_position_{window}'] = (df_features['close'] - lower_band) / (upper_band - lower_band)
            
            # 布林带突破
            df_features[f'bb_upper_break_{window}'] = 0
            df_features[f'bb_lower_break_{window}'] = 0
            df_features.loc[df_features['close'] > upper_band, f'bb_upper_break_{window}'] = 1
            df_features.loc[df_features['close'] < lower_band, f'bb_lower_break_{window}'] = 1
        
        # 添加到特征列表
        for window in Config.BB_WINDOWS:
            new_features = [f'bb_upper_{window}', f'bb_middle_{window}', f'bb_lower_{window}',
                           f'bb_bandwidth_{window}', f'bb_position_{window}', 
                           f'bb_upper_break_{window}', f'bb_lower_break_{window}']
            self.feature_names.extend(new_features)
        
        return df_features
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算波动率特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了波动率特征的数据
        """
        df_features = df.copy()
        
        for window in Config.VOLATILITY_WINDOWS:
            # 直接使用talib计算ATR
            df_features[f'atr_{window}'] = talib.ATR(
                df_features['high'].values,
                df_features['low'].values,
                df_features['close'].values,
                timeperiod=window
            )
            
            # 计算ATR百分比
            df_features[f'atr_pct_{window}'] = df_features[f'atr_{window}'] / df_features['close']
        
        # 添加到特征列表
        for window in Config.VOLATILITY_WINDOWS:
            self.feature_names.extend([f'atr_{window}', f'atr_pct_{window}'])
        
        return df_features
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算动量特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了动量特征的数据
        """
        df_features = df.copy()
        
        for window in Config.MOMENTUM_WINDOWS:
            # 直接使用talib计算动量指标和变化率
            # 动量指标
            df_features[f'momentum_{window}'] = talib.MOM(df_features['close'].values, timeperiod=window)
            # 变化率
            df_features[f'roc_{window}'] = talib.ROC(df_features['close'].values, timeperiod=window)
            
            # 计算动量百分比
            df_features[f'momentum_pct_{window}'] = df_features[f'momentum_{window}'] / df_features['close'].shift(window)
            
            # 计算相对强弱
            df_features[f'rs_{window}'] = df_features['close'] / df_features['close'].shift(window)
        
        # 直接使用talib计算威廉指标 (W%R)
        df_features['williams_r'] = talib.WILLR(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        # 添加到特征列表
        for window in Config.MOMENTUM_WINDOWS:
            self.feature_names.extend([f'momentum_{window}', f'momentum_pct_{window}', 
                                     f'roc_{window}', f'rs_{window}'])
        
        self.feature_names.append('williams_r')
        
        return df_features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算成交量特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了成交量特征的数据
        """
        df_features = df.copy()
        
        # 直接使用talib计算成交量移动平均
        for window in Config.VOLUME_PRICE_WINDOWS:
            df_features[f'volume_ma_{window}'] = talib.SMA(df_features['volume'].values, timeperiod=window)
            df_features[f'volume_ema_{window}'] = talib.EMA(df_features['volume'].values, timeperiod=window)
            
            # 计算成交量与均线的偏差
            df_features[f'volume_diff_pct_{window}'] = (df_features['volume'] - df_features[f'volume_ma_{window}']) / df_features[f'volume_ma_{window}']
        
        # 计算成交量加权平均价格 (VWAP)
        df_features['vwap'] = (df_features['volume'] * df_features['close']).rolling(window=10).sum() / df_features['volume'].rolling(window=10).sum()
        
        # 添加到特征列表
        for window in Config.VOLUME_PRICE_WINDOWS:
            self.feature_names.extend([f'volume_ma_{window}', f'volume_ema_{window}', f'volume_diff_pct_{window}'])
        self.feature_names.append('vwap')
        
        return df_features
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用talib计算统计特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了统计特征的数据
        """
        df_features = df.copy()
        
        for window in Config.STAT_WINDOWS:
            # 直接使用talib计算波动率
            df_features[f'volatility_{window}'] = talib.STDDEV(df_features['close'].values, timeperiod=window)
            
            # 计算偏度和峰度
            df_features[f'skew_{window}'] = df_features['close'].rolling(window=window).skew()
            df_features[f'kurtosis_{window}'] = df_features['close'].rolling(window=window).kurt()
            
            # 直接使用talib计算最高价和最低价
            df_features[f'high_{window}'] = talib.MAX(df_features['high'].values, timeperiod=window)
            df_features[f'low_{window}'] = talib.MIN(df_features['low'].values, timeperiod=window)
            
            # 计算价格范围
            df_features[f'price_range_{window}'] = df_features[f'high_{window}'] - df_features[f'low_{window}']
            df_features[f'price_range_pct_{window}'] = df_features[f'price_range_{window}'] / df_features['close']
            
            # 线性回归斜率 - 修复长度不匹配问题
            for i in range(window):
                df_features[f'lag_{i}'] = df_features['close'].shift(i)
            df_features[f'slope_{window}'] = df_features[[f'lag_{i}' for i in range(window)]].apply(
                lambda x: np.polyfit(range(len(x.dropna())), x.dropna(), 1)[0] if len(x.dropna()) >= 2 else np.nan, axis=1
            )
            # 删除滞后列
            for i in range(window):
                if f'lag_{i}' in df_features.columns:
                    df_features = df_features.drop(f'lag_{i}', axis=1)
        
        # 添加到特征列表
        for window in Config.STAT_WINDOWS:
            new_features = [f'skew_{window}', f'kurtosis_{window}', f'high_rolling_{window}',
                           f'low_rolling_{window}', f'price_range_rolling_{window}', f'slope_{window}']
            self.feature_names.extend(new_features)
        
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了滞后特征的数据
        """
        df_features = df.copy()
        
        # 价格滞后
        for lag in Config.LAG_FEATURES:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'return_lag_{lag}'] = df_features['return_1'].shift(lag)
        
        # 波动率滞后
        if 'realized_vol_5' in df_features.columns:
            df_features[f'vol_lag_1'] = df_features['realized_vol_5'].shift(1)
        
        # 添加到特征列表
        for lag in Config.LAG_FEATURES:
            new_features = [f'close_lag_{lag}', f'return_lag_{lag}']
            self.feature_names.extend(new_features)
        if 'vol_lag_1' in df_features.columns:
            self.feature_names.append('vol_lag_1')
        
        return df_features
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建基于时间的特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加了时间特征的数据
        """
        df_features = df.copy()
        
        # 确保索引是DatetimeIndex
        if isinstance(df_features.index, pd.DatetimeIndex):
            # 时间特征
            df_features['hour'] = df_features.index.hour
            df_features['minute'] = df_features.index.minute
            df_features['day_of_week'] = df_features.index.dayofweek
            df_features['day_of_month'] = df_features.index.day
            
            # 交易时段标识
            df_features['morning_session'] = ((df_features['hour'] >= 9) & (df_features['hour'] < 11)) | \
                                           ((df_features['hour'] == 11) & (df_features['minute'] <= 30))
            df_features['afternoon_session'] = (df_features['hour'] >= 13) & (df_features['hour'] < 15)
            
            # 添加到特征列表
            new_features = ['hour', 'minute', 'day_of_week', 'day_of_month', 'morning_session', 'afternoon_session']
            self.feature_names.extend(new_features)
        
        return df_features
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        标准化特征
        
        Args:
            df: 数据
            feature_cols: 特征列名列表
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (标准化后的数据, 标准化参数)
        """
        df_normalized = df.copy()
        norm_params = {}
        
        for col in feature_cols:
            if col in df_normalized.columns:
                # 计算均值和标准差
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                
                # 避免除零错误
                if std_val > 0:
                    df_normalized[f'{col}_norm'] = (df_normalized[col] - mean_val) / std_val
                    norm_params[col] = (mean_val, std_val)
        
        return df_normalized, norm_params
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        移除高度相关的特征
        
        Args:
            df: 数据
            feature_cols: 特征列名列表
            
        Returns:
            List[str]: 筛选后的特征列名列表
        """
        # 确保所有特征都在DataFrame中
        valid_features = [col for col in feature_cols if col in df.columns]
        
        # 只保留数值类型的特征
        numeric_features = []
        for col in valid_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                self.logger.debug(f"跳过非数值特征: {col}")
        
        # 计算相关系数矩阵（如果有足够的数值特征）
        if len(numeric_features) < 2:
            self.logger.warning(f"数值特征不足2个，无法计算相关性，保留所有特征")
            return valid_features
        
        # 计算相关系数矩阵
        corr_matrix = df[numeric_features].corr().abs()
        
        # 选择上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找到相关系数大于阈值的特征对
        to_drop = [column for column in upper.columns if any(upper[column] > Config.CORRELATION_THRESHOLD)]
        
        # 返回剩余特征
        selected_numeric_features = [col for col in numeric_features if col not in to_drop]
        
        # 添加回非数值特征
        non_numeric_features = [col for col in valid_features if col not in numeric_features]
        selected_features = selected_numeric_features + non_numeric_features
        
        self.logger.info(f"移除了 {len(to_drop)} 个高度相关的特征，保留 {len(selected_features)} 个特征")
        return selected_features
    
    def engineer_all_features(self, df: pd.DataFrame, target: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str], Optional[pd.DataFrame]]:
        """
        执行完整的特征工程流程
        
        Args:
            df: 原始数据（可能包含目标变量'target'列）
            target: 目标变量（可选，优先使用此参数）
            
        Returns:
            Tuple[pd.DataFrame, List[str], Optional[pd.DataFrame]]: 
                (添加了所有特征的数据, 特征列名列表, 处理后的目标变量)
        """
        self.logger.info("开始特征工程流程")
        original_index = df.index.copy()
        
        # 重置特征名称列表，确保不会重复
        self.feature_names = []
        
        # 复制数据以避免修改原始数据
        df_copy = df.copy()
        
        # 保存目标变量（如果存在）
        has_target = False
        if 'target' in df_copy.columns:
            target = df_copy['target'].copy()
            has_target = True
            # 从特征计算中移除目标变量，但在最后保留
            df_features = df_copy.drop('target', axis=1)
        else:
            df_features = df_copy.copy()
        
        # 1. 基础价格特征
        df_features = self.calculate_basic_price_features(df_features)
        
        # 2. 移动平均线
        df_features = self.calculate_moving_averages(df_features)
        
        # 3. RSI指标
        df_features = self.calculate_rsi(df_features)
        
        # 4. MACD指标
        df_features = self.calculate_macd(df_features)
        
        # 5. 布林带
        df_features = self.calculate_bollinger_bands(df_features)
        
        # 6. 波动率特征
        df_features = self.calculate_volatility_features(df_features)
        
        # 7. 动量特征
        df_features = self.calculate_momentum_features(df_features)
        
        # 8. 成交量特征
        df_features = self.calculate_volume_features(df_features)
        
        # 9. 统计特征
        df_features = self.calculate_statistical_features(df_features)
        
        # 10. 滞后特征
        df_features = self.create_lag_features(df_features)
        
        # 11. 时间特征
        df_features = self.create_time_based_features(df_features)
        
        # 清理数据
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # 删除由于移动窗口计算产生的NaN行
        df_features = df_features.dropna()
        
        # 确保目标变量和特征数据索引对齐
        processed_target = None
        if target is not None:
            # 确保target也是DataFrame格式
            if isinstance(target, pd.Series):
                target = target.to_frame()
            # 对齐目标变量的索引
            common_index = df_features.index.intersection(target.index)
            df_features = df_features.loc[common_index]
            processed_target = target.loc[common_index]
        
        # 显式收集所有生成的特征名（排除原始列）
        original_columns = set(['open', 'high', 'low', 'close', 'volume'])
        all_columns = set(df_features.columns)
        self.feature_names = list(all_columns - original_columns)
        
        # 移除高度相关的特征
        self.feature_names = self.remove_highly_correlated_features(df_features, self.feature_names)
        
        # 限制特征数量
        if Config.MAX_FEATURES > 0 and len(self.feature_names) > Config.MAX_FEATURES:
            self.logger.warning(f"特征数量 ({len(self.feature_names)}) 超过限制，将保留前 {Config.MAX_FEATURES} 个")
            self.feature_names = self.feature_names[:Config.MAX_FEATURES]
        
        # 将目标变量添加回结果数据框
        if has_target and target is not None:
            # 确保索引对齐
            target = target.reindex(df_features.index)
            df_features['target'] = target
        
        self.logger.info(f"特征工程完成，生成 {len(self.feature_names)} 个特征")
        self.logger.info(f"特征工程后数据行数: {len(df_features)}, 原始数据行数: {len(original_index)}")
        
        return df_features, self.feature_names, processed_target


if __name__ == "__main__":
    # 测试特征工程器
    from data_loader import FutureDataLoader
    
    # 加载数据
    loader = FutureDataLoader()
    data_files = loader.find_data_files()
    
    if data_files:
        # 加载第一个文件进行测试
        df = loader.load_single_file(data_files[0])
        df = loader.clean_data(df)
        df = loader.generate_target(df)
        
        # 执行特征工程
        engineer = FeatureEngineer()
        df_with_features, feature_names, _ = engineer.engineer_all_features(df)
        
        print(f"\n=== 特征工程结果 ===")
        print(f"生成的特征数量: {len(feature_names)}")
        print(f"特征列表:")
        for i, feature in enumerate(feature_names[:10], 1):
            print(f"  {i}. {feature}")
        if len(feature_names) > 10:
            print(f"  ... 以及 {len(feature_names) - 10} 个其他特征")
        
        print(f"\n特征数据形状: {df_with_features.shape}")
        print("\n前5行特征数据:")
        print(df_with_features[feature_names[:5]].head())