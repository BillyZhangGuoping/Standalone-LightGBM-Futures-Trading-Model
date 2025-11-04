#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 配置文件

此文件包含所有模型参数、路径配置和超参数搜索范围
"""

import os
import numpy as np
from datetime import timedelta

# ==========================
# 基础配置
# ==========================
class BaseConfig:
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 数据路径配置
    DATA_DIR = os.path.join(PROJECT_ROOT, '../future_data')
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data_lgbm')
    
    # 输出路径配置
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models_lgbm')
    PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots_lgbm')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs_lgbm')
    BACKTEST_DIR = os.path.join(PROJECT_ROOT, 'backtest_results_lgbm')
    
    # 确保目录存在
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.PROCESSED_DATA_DIR, cls.MODELS_DIR, 
                        cls.PLOTS_DIR, cls.LOGS_DIR, cls.BACKTEST_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# ==========================
# 数据处理配置
# ==========================
class DataConfig(BaseConfig):
    # 数据加载参数
    FILE_PATTERNS = ['*.csv']  # 支持的文件模式
    DATE_COLUMN = 'datetime'  # 日期列名（自动检测）
    OHLC_COLS = ['open', 'high', 'low', 'close']  # OHLC列名
    VOLUME_COL = 'volume'  # 成交量列名
    
    # 时间范围过滤
    START_DATE = None  # 起始日期（None表示不限制）
    END_DATE = None  # 结束日期（None表示不限制）
    
    # 数据清洗参数
    MAX_MISSING_RATIO = 0.05  # 最大缺失比例
    MAX_GAP_MINUTES = 30  # 最大时间间隔（分钟）
    
    # 目标变量参数
    LOOKAHEAD_MINUTES = 3  # 预测未来3分钟
    TARGET_THRESHOLD_TYPE = 'dynamic'  # 'static' 或 'dynamic'
    STATIC_THRESHOLD_UP = 0.0005  # 静态上涨阈值
    STATIC_THRESHOLD_DOWN = -0.0005  # 静态下跌阈值
    DYNAMIC_WINDOW = 200  # 动态阈值的滚动窗口
    DYNAMIC_QUANTILE_UP = 0.75  # 上涨阈值分位数
    DYNAMIC_QUANTILE_DOWN = 0.25  # 下跌阈值分位数
    
    # 数据分割参数
    TRAIN_RATIO = 0.7  # 训练集比例
    VAL_RATIO = 0.15  # 验证集比例
    TEST_RATIO = 0.15  # 测试集比例
    
    # 交叉验证参数
    CV_SPLITS = 5  # 时间序列交叉验证折数
    EARLY_STOPPING_ROUNDS = 100  # 早停轮数

# ==========================
# 特征工程配置
# ==========================
class FeatureConfig(BaseConfig):
    # 技术指标参数
    # 移动平均线
    MA_WINDOWS = [5, 10, 20, 50]  # 均线窗口
    
    # RSI参数
    RSI_WINDOWS = [6, 14, 21]  # RSI窗口
    
    # MACD参数
    MACD_FAST = 12  # 快线周期
    MACD_SLOW = 26  # 慢线周期
    MACD_SIGNAL = 9  # 信号线周期
    
    # 布林带参数
    BB_WINDOWS = [20]  # 布林带窗口
    BB_STD = 2  # 标准差倍数
    
    # 波动率参数
    VOLATILITY_WINDOWS = [5, 10, 20]  # 波动率窗口
    
    # 动量参数
    MOMENTUM_WINDOWS = [5, 10, 20]  # 动量窗口
    
    # 量价关系参数
    VOLUME_PRICE_WINDOWS = [5, 10]  # 量价关系窗口
    
    # 统计特征参数
    STAT_WINDOWS = [10, 20, 50]  # 统计特征窗口
    
    # 滞后特征参数
    LAG_FEATURES = [1, 2, 3, 5, 10]  # 滞后步长
    
    # 特征选择参数
    FEATURE_IMPORTANCE_THRESHOLD = 0.001  # 特征重要性阈值
    MAX_FEATURES = 50  # 最大特征数量
    CORRELATION_THRESHOLD = 0.95  # 相关系数阈值

# ==========================
# 模型配置
# ==========================
class ModelConfig(BaseConfig):
    # LightGBM基础参数
    BOOSTER_TYPE = 'gbdt'  # 提升器类型
    OBJECTIVE = 'multiclass'  # 目标函数
    NUM_CLASS = 3  # 类别数量
    METRIC = 'multi_logloss'  # 评估指标
    VERBOSE = 100  # 日志输出频率
    
    # 训练参数
    NUM_BOOST_ROUND = 10000  # 最大迭代次数
    VALIDATION_FRACTION = 0.2  # 验证集比例
    
    # 类别平衡参数
    IS_UNBALANCE = True  # 是否处理不平衡数据
    
    # Optuna超参数搜索范围
    OPTUNA_SEARCH_SPACE = {
        'num_leaves': {'low': 16, 'high': 128, 'step': 8},
        'learning_rate': {'low': 0.005, 'high': 0.1, 'log': True},
        'feature_fraction': {'low': 0.6, 'high': 1.0},
        'bagging_fraction': {'low': 0.6, 'high': 1.0},
        'bagging_freq': {'low': 1, 'high': 10},
        'lambda_l1': {'low': 0.0, 'high': 10.0, 'log': True},
        'lambda_l2': {'low': 0.0, 'high': 10.0, 'log': True},
        'min_data_in_leaf': {'low': 10, 'high': 100, 'step': 5},
        'min_gain_to_split': {'low': 0.0, 'high': 1.0},
    }
    
    # Optuna优化参数
    OPTUNA_N_TRIALS = 100  # 试验次数
    OPTUNA_SEED = 42  # 随机种子
    
    # 模型保存参数
    MODEL_VERSION = 'v1.0'
    SAVE_BEST_ONLY = True
    SAVE_FEATURE_IMPORTANCE = True

# ==========================
# 回测配置
# ==========================
class BacktestConfig(BaseConfig):
    # 交易参数
    INITIAL_CAPITAL = 1000000.0  # 初始资金
    POSITION_SIZE_RATIO = 0.1  # 持仓比例
    COMMISSION_RATE = 0.0001  # 手续费率
    SLIPPAGE_RATE = 0.00005  # 滑点率
    
    # 信号过滤参数
    MIN_CONFIDENCE = 0.55  # 最小置信度
    SIGNALS_CONSENSUS = 2  # 信号一致性要求
    
    # 止损止盈参数
    TAKE_PROFIT_RATIO = 0.002  # 止盈比例
    STOP_LOSS_RATIO = 0.001  # 止损比例
    
    # 回测时间窗口
    SHORT_WINDOW = timedelta(days=1)  # 短期评估窗口
    MEDIUM_WINDOW = timedelta(weeks=1)  # 中期评估窗口
    LONG_WINDOW = timedelta(days=30)  # 长期评估窗口
    
    # 绩效评估参数
    RISK_FREE_RATE = 0.02  # 无风险收益率
    MAX_DRAWDOWN_THRESHOLD = 0.2  # 最大回撤阈值
    
    # 可视化参数
    PLOT_TRADES = True  # 是否绘制交易点
    PLOT_EQUITY_CURVE = True  # 是否绘制权益曲线
    PLOT_HEATMAP = True  # 是否绘制热力图

# ==========================
# 日志配置
# ==========================
class LogConfig(BaseConfig):
    LOG_LEVEL = 'INFO'  # 日志级别
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BaseConfig.LOGS_DIR, 'lgbm_trading_model.log')

# ==========================
# 合并配置
# ==========================
class Config:
    """合并所有配置"""
    
    # 初始化所有配置
    @classmethod
    def initialize(cls):
        # 确保所有目录存在
        BaseConfig.ensure_dirs()
        
        # 将子配置类的属性合并到主配置类
        for config_class in [BaseConfig, DataConfig, FeatureConfig, ModelConfig, BacktestConfig, LogConfig]:
            for attr_name, attr_value in config_class.__dict__.items():
                if not attr_name.startswith('__') and not callable(attr_value):
                    setattr(cls, attr_name, attr_value)

# 初始化配置
Config.initialize()

if __name__ == "__main__":
    # 打印配置信息
    print("=== 项目配置信息 ===")
    print(f"项目根目录: {Config.PROJECT_ROOT}")
    print(f"数据目录: {Config.DATA_DIR}")
    print(f"模型保存目录: {Config.MODELS_DIR}")
    print(f"预测未来时间: {Config.LOOKAHEAD_MINUTES}分钟")
    print(f"最大特征数量: {Config.MAX_FEATURES}")
    print(f"目标类别数: {Config.NUM_CLASS}")