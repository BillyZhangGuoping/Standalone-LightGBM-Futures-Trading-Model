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
    DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
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
    
    # 目标变量参数 - 优化以提高F1分数
    LOOKAHEAD_MINUTES = 1  # 减小预测窗口以增加交易信号数量
    TARGET_THRESHOLD_TYPE = 'dynamic'  # 切换到动态阈值以更好地适应市场变化
    STATIC_THRESHOLD_UP = 0.0002  # 进一步降低上涨阈值，增加上涨信号
    STATIC_THRESHOLD_DOWN = -0.0002  # 进一步降低下跌阈值，增加下跌信号
    DYNAMIC_WINDOW = 250  # 增加动态阈值窗口以捕获更长期的模式
    DYNAMIC_QUANTILE_UP = 0.72  # 略微降低上涨阈值分位数，增加上涨信号
    DYNAMIC_QUANTILE_DOWN = 0.28  # 略微提高下跌阈值分位数，增加下跌信号
    
    # 高级目标变量参数
    FOCUS_ON_MINORITY_CLASSES = True  # 重点关注少数类别
    ENABLE_CLASS_WEIGHTING = True  # 启用类别加权
    
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
    # 技术指标参数 - 扩展窗口以捕获更多模式
    MA_WINDOWS = [2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 150, 200]  # 进一步增加多种周期
    
    # RSI参数 - 增加更多周期以提高响应速度和稳定性
    RSI_WINDOWS = [2, 3, 5, 6, 9, 12, 14, 21, 30, 50]  # 增加更多RSI周期
    
    # MACD参数 - 保持经典参数
    MACD_FAST = 12  # 快线周期
    MACD_SLOW = 26  # 慢线周期
    MACD_SIGNAL = 9  # 信号线周期
    
    # 布林带参数 - 增加更多窗口和标准差选项
    BB_WINDOWS = [10, 20, 30, 50]  # 增加50周期
    BB_STD = [1.5, 2.0, 2.5]  # 增加多个标准差选项
    
    # 波动率参数 - 进一步扩展范围
    VOLATILITY_WINDOWS = [3, 5, 10, 20, 30, 60, 120]  # 增加120周期
    
    # 动量参数 - 增加更多时间窗口
    MOMENTUM_WINDOWS = [3, 5, 10, 20, 50]  # 增加50周期
    
    # 量价关系参数 - 增加更多窗口
    VOLUME_PRICE_WINDOWS = [5, 10, 20, 50]  # 增加50周期
    
    # 统计特征参数 - 增加更多窗口
    STAT_WINDOWS = [5, 10, 20, 50, 100]  # 增加100周期
    
    # 滞后特征参数 - 调整滞后步长以捕获更多时间模式
    LAG_FEATURES = [1, 3, 6, 12, 24, 48]  # 增加48周期以捕获日度模式
    
    # 特征选择参数 - 进一步放宽以允许更多潜在有用特征
    FEATURE_IMPORTANCE_THRESHOLD = 0.0003  # 进一步降低特征重要性阈值
    MAX_FEATURES = 120  # 进一步增加最大特征数量
    CORRELATION_THRESHOLD = 0.92  # 略微调整相关系数阈值
    ENABLE_RECURSIVE_FEATURE_ELIMINATION = True  # 启用递归特征消除

# ==========================
# 模型配置
# ==========================
class ModelConfig(BaseConfig):
    # LightGBM基础参数
    BOOSTER_TYPE = 'gbdt'  # 提升器类型
    OBJECTIVE = 'multiclass'  # 目标函数
    NUM_CLASS = 3  # 类别数量
    METRIC = 'multi_logloss'  # 评估指标 (内部计算)
    EVAL_METRIC = ['multi_logloss', 'multi_error', 'auc_mu']  # 增加多指标评估
    VERBOSE = 100  # 日志输出频率
    
    # 训练参数 - 增加最大迭代次数
    NUM_BOOST_ROUND = 8000  # 增加最大迭代次数以捕获更复杂的模式
    VALIDATION_FRACTION = 0.25  # 验证集比例
    
    # 类别平衡参数 - 增强以提高F1分数
    IS_UNBALANCE = True  # 启用自动不平衡处理
    ENABLE_RANDOM_SEED = True  # 启用随机种子保证可复现性
    RANDOM_SEED = 42  # 随机种子值
    CLASS_WEIGHTS = 'balanced'  # 使用balanced权重策略
    
    # SMOTE过采样配置
    SMOTE_ENABLED = True  # 启用SMOTE过采样
    SMOTE_SAMPLING_STRATEGY = 'auto'  # 采样策略
    SMOTE_K_NEIGHBORS = 5  # SMOTE的K近邻数
    
    # Optuna超参数搜索范围 - 优化以提高F1分数
    OPTUNA_SEARCH_SPACE = {
        'num_leaves': {'low': 32, 'high': 384, 'step': 16},  # 进一步增加上限以提高模型复杂度
        'learning_rate': {'low': 0.001, 'high': 0.08, 'log': True},  # 扩大学习率范围上限
        'feature_fraction': {'low': 0.3, 'high': 1.0},  # 特征采样范围
        'bagging_fraction': {'low': 0.5, 'high': 1.0},  # 数据采样范围
        'bagging_freq': {'low': 1, 'high': 30},  # 采样频率范围
        'lambda_l1': {'low': 0.0001, 'high': 15.0, 'log': True},  # 进一步扩大L1正则化范围
        'lambda_l2': {'low': 0.0001, 'high': 15.0, 'log': True},  # 进一步扩大L2正则化范围
        'min_data_in_leaf': {'low': 8, 'high': 250, 'step': 10},  # 进一步扩大叶节点最小数据量范围
        'min_gain_to_split': {'low': 0.000001, 'high': 0.2},  # 进一步扩大最小分裂增益阈值范围
        'max_depth': {'low': 8, 'high': 35},  # 扩大树深度搜索范围
        'scale_pos_weight': {'low': 0.3, 'high': 3.0},  # 扩大正负样本权重比例参数范围
        'extra_trees': {'low': 0, 'high': 1, 'step': 1},  # 添加额外树参数
        'max_bin': {'low': 128, 'high': 512, 'step': 64},  # 添加最大分箱参数
    }
    
    # 过拟合控制参数
    MAX_DEPTH = 25  # 增加树深度以提高模型容量
    EARLY_STOPPING_ROUNDS = 200  # 增加早停轮数以允许更充分的训练
    FEATURE_PRE_FILTER = False  # 禁用特征预过滤，允许动态调整min_data_in_leaf
    
    # Optuna优化参数 - 增强以提高F1分数
    OPTUNA_N_TRIALS = 400  # 进一步增加试验次数
    OPTUNA_SEED = 42  # 随机种子
    OPTUNA_SAMPLER = 'TPESampler'  # 使用Tree-structured Parzen Estimator
    OPTUNA_SAMPLER_MULTIVARIATE = True  # 启用多变量采样
    OPTUNA_N_STARTUP_TRIALS = 40  # 增加初始随机采样次数
    OPTUNA_PRUNER = 'HyperbandPruner'  # 使用Hyperband剪枝加速优化
    OPTUNA_SCORE_FUNC = 'f1_weighted'  # 使用加权F1分数作为优化目标
    OPTUNA_DIRECTION = 'maximize'  # 最大化目标函数
    
    # 模型集成参数
    BAGGING_FRACTION = 0.8  # 数据采样比例
    BAGGING_FREQUENCY = 1  # 数据采样频率
    FEATURE_FRACTION = 0.8  # 特征采样比例
    
    # 模型保存参数
    MODEL_VERSION = 'v1.0'
    SAVE_BEST_ONLY = True
    SAVE_FEATURE_IMPORTANCE = True

# ==========================
# 回测配置
# ==========================
class BacktestConfig(BaseConfig):
    # 交易参数
    INITIAL_CAPITAL = 10000.0  # 初始资金
    POSITION_SIZE_RATIO = 0.1  # 持仓比例
    COMMISSION_RATE = 0.0001  # 手续费率
    SLIPPAGE_RATE = 0.00005  # 滑点率
    
    # 信号过滤参数
    MIN_CONFIDENCE = 0.45  # 降低最小置信度以接受更多交易信号
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
    MAX_DRAWDOWN_THRESHOLD = 0.3  # 最大回撤阈值
    
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