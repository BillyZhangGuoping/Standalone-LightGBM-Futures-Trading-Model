#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 数据加载器

此模块负责加载、清洗和预处理期货数据，并生成目标变量
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config import Config

# 设置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_loader')


class FutureDataLoader:
    """
    期货数据加载器，负责数据的加载、清洗和预处理
    """
    
    def __init__(self, data_dir=None):
        """初始化数据加载器"""
        self.data_dir = data_dir if data_dir is not None else Config.DATA_DIR
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
        self.logger = logger
        self.logger.info(f"数据加载器初始化完成，数据目录: {self.data_dir}")
    
    def find_data_files(self, symbol=None) -> List[str]:
        """
        查找数据文件，可选择性地按品种过滤
        
        Args:
            symbol: 可选的交易品种代码，用于过滤文件
            
        Returns:
            List[str]: 数据文件路径列表
        """
        data_files = []
        for pattern in Config.FILE_PATTERNS:
            search_path = os.path.join(self.data_dir, pattern)
            files = glob.glob(search_path)
            # 如果指定了symbol，则过滤文件名中包含该symbol的文件
            if symbol:
                files = [f for f in files if symbol in os.path.basename(f)]
            data_files.extend(files)
        
        self.logger.info(f"找到 {len(data_files)} 个数据文件")
        return sorted(data_files)
    
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        加载单个数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            self.logger.info(f"加载文件: {file_path}")
            df = pd.read_csv(file_path)
            
            # 检测并处理日期列
            df = self._process_date_column(df)
            
            # 检测并处理OHLC列
            df = self._ensure_ohlc_columns(df)
            
            # 排序
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            
            self.logger.info(f"文件加载完成，形状: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            raise
    
    def _process_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理日期列
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 尝试常见的日期列名
        date_columns = ['datetime', 'date_time', 'timestamp', 'time', 'date', 'bob']
        date_col_found = None
        
        # 检查是否有现成的日期列
        for col in date_columns:
            if col in df.columns:
                # 如果同时有date和time列，合并它们
                if col == 'date' and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                    df = df.drop(['date', 'time'], axis=1)
                    date_col_found = 'datetime'
                    break
                else:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col_found = col
                        break
                    except:
                        continue
        
        # 如果找到日期列，设置为索引
        if date_col_found:
            df = df.set_index(date_col_found)
            self.logger.debug(f"使用 '{date_col_found}' 作为日期索引")
        else:
            # 如果没有找到日期列，创建一个
            self.logger.warning("未找到有效的日期列，创建基于索引的时间戳")
            start_time = datetime.now() - timedelta(minutes=len(df))
            df.index = [start_time + timedelta(minutes=i) for i in range(len(df))]
        
        return df
    
    def _ensure_ohlc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保OHLC列存在
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 检查OHLC列
        for col in Config.OHLC_COLS:
            if col not in df.columns:
                # 尝试常见的替代名称
                alternatives = {
                    'open': ['开盘价', 'open_price', 'o'],
                    'high': ['最高价', 'high_price', 'h'],
                    'low': ['最低价', 'low_price', 'l'],
                    'close': ['收盘价', 'close_price', 'c', 'price']
                }
                
                found = False
                for alt in alternatives.get(col, []):
                    if alt in df.columns:
                        df = df.rename(columns={alt: col})
                        self.logger.debug(f"将 '{alt}' 重命名为 '{col}'")
                        found = True
                        break
                
                # 如果找不到替代名称且是收盘价，使用价格列或创建默认值
                if not found and col == 'close' and len(df.columns) > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        df[col] = df[numeric_cols[0]]
                        self.logger.warning(f"使用 '{numeric_cols[0]}' 作为收盘价")
                    else:
                        df[col] = 0
                        self.logger.warning(f"创建默认的 '{col}' 列")
                elif not found:
                    df[col] = 0
                    self.logger.warning(f"创建默认的 '{col}' 列")
        
        # 检查成交量列
        if Config.VOLUME_COL not in df.columns:
            volume_alts = ['成交量', 'vol', 'volume_traded']
            found = False
            for alt in volume_alts:
                if alt in df.columns:
                    df = df.rename(columns={alt: Config.VOLUME_COL})
                    self.logger.debug(f"将 '{alt}' 重命名为 '{Config.VOLUME_COL}'")
                    found = True
                    break
            if not found:
                df[Config.VOLUME_COL] = 0
                self.logger.warning(f"创建默认的 '{Config.VOLUME_COL}' 列")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        验证数据质量
        
        Args:
            df: 待验证的数据
            
        Returns:
            Tuple[bool, str]: (是否通过验证, 消息)
        """
        # 检查缺失值
        missing_ratio = df.isnull().mean().mean()
        if missing_ratio > Config.MAX_MISSING_RATIO:
            return False, f"缺失值比例过高: {missing_ratio:.4f}"
        
        # 检查时间连续性
        if isinstance(df.index, pd.DatetimeIndex):
            # 计算时间间隔
            gaps = df.index.to_series().diff().dropna()
            max_gap = gaps.max()
            if max_gap > timedelta(minutes=Config.MAX_GAP_MINUTES):
                return False, f"时间间隔过大: {max_gap}"
        
        # 检查OHLC值
        for col in Config.OHLC_COLS:
            if col in df.columns:
                if (df[col] <= 0).any():
                    return False, f"{col} 列包含无效值"
        
        return True, f"数据质量验证通过，缺失值比例: {missing_ratio:.6f}"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        # 复制数据以避免修改原始数据
        df_clean = df.copy()
        
        # 处理缺失值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 使用前向填充和后向填充
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            # 如果仍然有缺失值，使用均值填充
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        # 处理异常值（使用IQR方法）
        for col in numeric_cols:
            if col in Config.OHLC_COLS + [Config.VOLUME_COL]:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 5 * IQR  # 放宽限制以避免过度过滤
                upper_bound = Q3 + 5 * IQR
                
                # 将异常值限制在边界内
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 确保OHLC逻辑正确
        if all(col in df_clean.columns for col in ['high', 'low', 'open', 'close']):
            # 确保high >= open/close/low
            df_clean['high'] = df_clean[['high', 'open', 'close', 'low']].max(axis=1)
            # 确保low <= open/close/high
            df_clean['low'] = df_clean[['low', 'open', 'close', 'high']].min(axis=1)
        
        # 重新索引以确保时间连续性
        if isinstance(df_clean.index, pd.DatetimeIndex):
            # 生成完整的分钟级时间索引
            full_index = pd.date_range(
                start=df_clean.index.min(), 
                end=df_clean.index.max(), 
                freq='min'
            )
            # 重新索引并填充缺失值
            df_clean = df_clean.reindex(full_index)
            # 再次填充缺失值
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"数据清洗完成，清洗后形状: {df_clean.shape}")
        return df_clean
    
    def generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成目标变量
        
        Args:
            df: 清洗后的数据
            
        Returns:
            pd.DataFrame: 添加了目标变量的数据
        """
        # 计算未来收益率
        df_target = df.copy()
        if 'close' in df_target.columns:
            # 计算未来3分钟的收益率
            df_target['future_close'] = df_target['close'].shift(-Config.LOOKAHEAD_MINUTES)
            df_target['future_return'] = (df_target['future_close'] - df_target['close']) / df_target['close']
            
            # 根据配置选择阈值类型
            if Config.TARGET_THRESHOLD_TYPE == 'static':
                # 静态阈值
                df_target['target'] = 0  # 默认为震荡
                df_target.loc[df_target['future_return'] > Config.STATIC_THRESHOLD_UP, 'target'] = 1  # 上涨
                df_target.loc[df_target['future_return'] < Config.STATIC_THRESHOLD_DOWN, 'target'] = -1  # 下跌
            else:
                # 动态阈值（基于历史收益率的滚动分位数）
                # 计算历史收益率，避免使用未来数据
                df_target['historical_return'] = df_target['close'].pct_change()
                roll_window = Config.DYNAMIC_WINDOW
                
                # 使用历史收益率的滚动分位数作为阈值
                # min_periods=1确保即使在窗口不够的情况下也能计算
                df_target['rolling_q_up'] = df_target['historical_return'].rolling(
                    window=roll_window, min_periods=1).quantile(Config.DYNAMIC_QUANTILE_UP)
                df_target['rolling_q_down'] = df_target['historical_return'].rolling(
                    window=roll_window, min_periods=1).quantile(Config.DYNAMIC_QUANTILE_DOWN)
                
                # 使用动态阈值分类
                df_target['target'] = 0  # 默认为震荡
                df_target.loc[df_target['future_return'] > df_target['rolling_q_up'], 'target'] = 1  # 上涨
                df_target.loc[df_target['future_return'] < df_target['rolling_q_down'], 'target'] = -1  # 下跌
                
                # 填充边界值
                df_target['target'] = df_target['target'].fillna(0)
            
            # 删除用于计算的中间列
            df_target = df_target.drop(['future_close', 'future_return', 'historical_return', 'rolling_q_up', 'rolling_q_down'], axis=1, errors='ignore')
            
            # 统计目标分布
            target_dist = df_target['target'].value_counts().sort_index()
            self.logger.info(f"目标变量分布: {target_dist.to_dict()}")
            
            # 计算类别权重
            total_samples = len(df_target)
            class_weights = {}
            for target_class in [-1, 0, 1]:
                count = target_dist.get(target_class, 0)
                class_weights[target_class] = total_samples / (3 * count) if count > 0 else 1.0
            self.logger.info(f"计算的类别权重: {class_weights}")
            
            # 存储类别权重
            df_target['class_weight'] = df_target['target'].map(class_weights)
        
        return df_target
    
    def load_and_preprocess_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        加载并预处理所有文件
        
        Returns:
            Dict[str, pd.DataFrame]: 处理后的数据字典
        """
        processed_data = {}
        data_files = self.find_data_files()
        
        for file_path in data_files:
            try:
                # 获取文件名作为标识符
                symbol = os.path.splitext(os.path.basename(file_path))[0]
                self.logger.info(f"处理品种: {symbol}")
                
                # 加载数据
                df = self.load_single_file(file_path)
                
                # 验证数据质量
                is_valid, msg = self.validate_data_quality(df)
                if not is_valid:
                    self.logger.warning(f"数据质量验证失败: {msg}，跳过该文件")
                    continue
                
                # 清洗数据
                df_clean = self.clean_data(df)
                
                # 生成目标变量
                df_final = self.generate_target(df_clean)
                
                # 保存处理后的数据
                processed_data[symbol] = df_final
                
                # 保存到文件
                output_file = os.path.join(self.processed_data_dir, f"{symbol}_processed.parquet")
                df_final.to_parquet(output_file)
                self.logger.info(f"处理后的数据已保存至: {output_file}")
                
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        
        self.logger.info(f"所有文件处理完成，成功处理 {len(processed_data)} 个文件")
        return processed_data
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分割数据集
        
        Args:
            df: 完整数据集
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (训练集, 验证集, 测试集)
        """
        # 移除NAN值
        df = df.dropna(subset=['target'])
        
        # 计算分割点
        n_samples = len(df)
        train_size = int(n_samples * Config.TRAIN_RATIO)
        val_size = int(n_samples * Config.VAL_RATIO)
        
        # 分割数据（保持时间顺序）
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        self.logger.info(f"数据分割完成: 训练集 {len(train_df)} 样本, 验证集 {len(val_df)} 样本, 测试集 {len(test_df)} 样本")
        
        # 检查目标分布
        self.logger.info("训练集目标分布:")
        for target, count in train_df['target'].value_counts().items():
            self.logger.info(f"  类别 {target}: {count} ({count/len(train_df)*100:.2f}%)")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # 测试数据加载器
    loader = FutureDataLoader()
    processed_data = loader.load_and_preprocess_all_files()
    
    if processed_data:
        # 显示第一个数据的信息
        first_symbol, first_data = next(iter(processed_data.items()))
        print(f"\n=== {first_symbol} 数据概览 ===")
        print(f"数据形状: {first_data.shape}")
        print(f"时间范围: {first_data.index.min()} 到 {first_data.index.max()}")
        print(f"目标分布:\n{first_data['target'].value_counts().sort_index()}")
        print("\n前5行数据:")
        print(first_data.head())