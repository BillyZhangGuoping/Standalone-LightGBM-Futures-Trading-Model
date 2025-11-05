# LightGBM期货交易信号预测模型

## 项目概述

本项目实现了一个基于LightGBM的期货交易信号预测系统，目标是预测未来N个周期后的价格方向（大涨/大跌/震荡）。系统采用了完整的机器学习工作流程，包括数据预处理、特征工程、模型训练、超参数优化、交叉验证和交易回测。

### 主要功能

- **多品种数据处理**：支持读取和处理多个期货品种的1分钟K线数据
- **高级特征工程**：生成50+个技术指标和统计特征
- **智能目标定义**：使用动态阈值对未来收益率进行三分类
- **超参数优化**：基于Optuna的自动参数搜索
- **时间序列交叉验证**：确保模型在时间维度上的稳健性
- **完整交易回测**：支持止损止盈、交易成本、滑点等真实交易因素
- **全面性能分析**：包括分类指标和交易指标的综合评估
- **可视化和报告**：自动生成图表和详细的性能报告

## 系统架构

系统采用模块化设计，各组件之间松耦合，便于维护和扩展：

```
├── data_loader.py    # 数据加载和预处理模块
├── feature_engineer.py  # 特征工程模块
├── model_trainer.py  # 模型训练和评估模块
├── backtester.py     # 交易回测和性能分析模块
├── config.py         # 配置管理模块
├── main.py           # 主程序，整合各模块
├── requirements.txt  # 依赖库列表
└── README.md         # 项目文档
```

## 安装指南

### 环境要求

- Python 3.8+
- pip 20.0+

### 安装步骤

1. 克隆或下载项目代码到本地

2. 安装依赖库：

```bash
cd  LightGBM_Prediction_Singal
pip install -r requirements.txt
```

3. 创建必要的目录结构：

```bash
mkdir -p raw_data processed_data models plots backtest_results logs
```

## 使用说明

### 数据准备

1. 将期货K线数据（CSV格式）放入`raw_data`目录
2. 确保数据包含以下列：`datetime`, `open`, `high`, `low`, `close`, `volume`

### 运行模型

基本用法：

```bash
python main.py
```

命令行参数：

```
--data-dir          # 数据目录路径，默认: data/
--symbol            # 特定交易品种代码，不指定则使用所有品种
--model-version     # 模型版本号，默认: v1
--optimize          # 是否进行超参数优化，默认: True
--n-trials          # Optuna优化的试验次数，默认: 100
--backtest          # 是否进行回测，默认: True
--signal-threshold  # 信号阈值，默认: 0.0
--save-results      # 是否保存结果，默认: True
--plot-results      # 是否绘制结果图表，默认: True
--balanced-weight   # 是否使用平衡权重，默认: True
--check-leakage     # 是否检查验证集数据泄漏，默认: True
```

示例：

```bash
# 使用特定品种，减少优化试验次数以加快速度
python main.py --symbol fu2601 --n-trials 20 --signal-threshold 0.2
```

## 模块说明

### 1. 配置模块 (config.py)

集中管理所有配置参数：
- 数据处理参数：目标窗口、阈值设置、数据分割比例
- 特征工程参数：技术指标窗口、滞后特征数量
- 模型参数：LightGBM配置、Optuna搜索空间
- 回测参数：初始资金、交易成本、止损止盈设置

### 2. 数据加载模块 (data_loader.py)

负责数据的读取、清洗和预处理：
- 自动查找和加载数据文件
- 数据质量验证和缺失值处理
- 异常值检测和修复
- 目标变量生成（动态阈值分类）
- 时间序列数据分割

### 3. 特征工程模块 (feature_engineer.py)

生成丰富的技术和统计特征：
- 基础价格特征：OHLC变换、收益率
- 技术指标：RSI、MACD、布林带等
- 统计特征：滚动均值、标准差、偏度等
- 滞后特征：价格和收益率的历史值
- 特征选择和相关性分析

### 4. 模型训练模块 (model_trainer.py)

实现模型的训练、优化和评估：
- LightGBM模型配置和训练
- 支持平衡权重（balanced class weight）处理不平衡数据
- 验证集数据泄漏检查功能
- Optuna超参数优化
- 时间序列交叉验证
- 模型评估和特征重要性分析
- 模型保存和加载

### 5. 回测模块 (backtester.py)

模拟交易策略并评估性能：
- 基于模型预测的交易信号生成
- 完整的交易执行模拟（开平仓、成本计算）
- 止损止盈策略实现
- 性能指标计算（收益率、夏普比率、最大回撤等）
- 可视化和报告生成

### 模型训练方法参数说明

`LightGBMTrainer`类的`train_model`方法支持以下关键参数：

- **X_train, y_train**: 训练集特征和目标变量
- **X_val, y_val**: 验证集特征和目标变量
- **feature_names**: 特征名称列表
- **optimize**: 是否进行超参数优化，默认True
- **use_balanced_weight**: 是否使用平衡权重处理不平衡数据，默认True
  - 当设置为True时，系统会根据类别分布自动计算和应用权重
  - 有助于提高少数类别的预测性能，减少类别不平衡的影响

### 数据泄漏检查功能

系统自动执行验证集数据泄漏检查：
- 检测训练集和验证集之间的重复数据
- 确保验证集数据不会在训练过程中被意外使用
- 在发现潜在泄漏时提供详细警告日志
- 有助于提高模型评估的可靠性和泛化能力

## 配置指南

主要配置参数位于`config.py`文件中，可根据需要调整：

### 数据处理配置

```python
# 目标变量定义
LOOKAHEAD_MINUTES = 3  # 预测未来3分钟
TARGET_THRESHOLD_TYPE = 'dynamic'  # 'static' 或 'dynamic'
STATIC_THRESHOLD_UP = 0.0005  # 静态上涨阈值
STATIC_THRESHOLD_DOWN = -0.0005  # 静态下跌阈值
DYNAMIC_WINDOW = 200  # 动态阈值的滚动窗口
DYNAMIC_QUANTILE_UP = 0.75  # 上涨阈值分位数
DYNAMIC_QUANTILE_DOWN = 0.25  # 下跌阈值分位数

# 数据分割
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.15  # 验证集比例
TEST_RATIO = 0.15  # 测试集比例
```

### 模型训练配置

```python
# LightGBM参数
BOOSTER_TYPE = 'gbdt'
OBJECTIVE = 'multiclass'
NUM_CLASS = 3
METRIC = 'multi_logloss'
VERBOSE = 100
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 100
IS_UNBALANCE = True  # 可以通过use_balanced_weight参数调整

# Optuna超参数优化
OPTUNA_N_TRIALS = 100
OPTUNA_SEED = 42
MODEL_VERSION = 'v1.0'
```

### 回测配置

```python
# 回测参数
INITIAL_CAPITAL = 1000000.0  # 初始资金
POSITION_SIZE_RATIO = 0.1  # 持仓比例
COMMISSION_RATE = 0.0001  # 手续费率
SLIPPAGE_RATE = 0.00005  # 滑点率
TAKE_PROFIT_RATIO = 0.002  # 止盈比例
STOP_LOSS_RATIO = 0.001  # 止损比例
MIN_CONFIDENCE = 0.55  # 最小置信度
```

## 性能指标解释

### 分类性能指标

- **精确率(Precision)**：预测为正例的样本中实际正例的比例
- **召回率(Recall)**：实际正例被预测为正例的比例
- **F1分数**：精确率和召回率的调和平均
- **混淆矩阵**：展示预测类别与实际类别的对应关系
- **ROC-AUC**：曲线下面积，衡量模型区分不同类别的能力

### 交易性能指标

- **总收益率**：投资期间的总回报百分比
- **年化收益率**：年化后的收益率，便于比较不同时间长度的策略
- **夏普比率**：每单位风险的超额回报，通常>1为好
- **最大回撤**：投资期间的最大亏损幅度
- **卡尔玛比率**：年化收益率与最大回撤的比值
- **胜率**：盈利交易占总交易的比例
- **盈亏比**：平均盈利与平均亏损的比值

## 结果输出

### 模型输出

- 训练好的模型文件：`models_lgbm/lgbm_model_{version}_{timestamp}.txt`
- 模型元数据：`models_lgbm/lgbm_model_{version}_{timestamp}_metadata.pkl`

### 可视化输出

- 特征重要性：`plots_lgbm/feature_importance.png`
- 学习曲线：`plots_lgbm/learning_curve.png`
- 混淆矩阵：`plots_lgbm/confusion_matrix.png`

### 回测输出

- 资产价值曲线：`backtest_results_lgbm/asset_value.png`
- 回撤曲线：`backtest_results_lgbm/drawdown.png`
- 信号和持仓图表：`backtest_results_lgbm/signals_and_positions.png`
- 交易分布：`backtest_results_lgbm/trade_distributions.png`
- 性能报告：`backtest_results_lgbm/performance_report_{timestamp}.txt`
- 交易历史：`backtest_results_lgbm/trade_history_{timestamp}.csv`

## 注意事项和最佳实践

1. **数据质量**：确保输入数据的质量，包括时间戳连续性和OHLC数据的有效性
2. **过拟合防范**：使用交叉验证和早停机制避免过拟合
3. **参数调整**：根据不同品种和市场环境调整阈值和回测参数
4. **实时监控**：在实盘应用前，建议长时间的离线测试和监控
5. **风险控制**：始终设置合理的止损，控制单笔交易的最大风险
6. **定期重训**：市场环境变化时，定期重新训练模型以保持预测准确性

## 常见问题

### Q: 模型预测准确率较低怎么办？
A: 可以尝试以下方法：
- 增加更多的特征或调整现有特征
- 调整目标变量的阈值
- 增加Optuna优化的试验次数
- 尝试不同的模型参数或其他模型

### Q: 回测结果显示大量交易怎么办？
A: 可以提高信号阈值或调整模型参数以减少交易频率

### Q: 如何处理不同品种的数据？
A: 系统支持多品种数据处理，但建议分别训练模型以适应不同品种的特性

## 许可证

本项目仅供研究和学习使用，不构成投资建议。

## 联系方式

- 项目维护者：[Billy, Zhang Guoping]
- 邮箱：[zhanggp@gmail.com]

如有问题或建议，请随时提出反馈。