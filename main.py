#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 主程序

此程序整合了所有模块，提供完整的工作流程：
1. 数据加载和预处理
2. 特征工程
3. 模型训练和超参数优化
4. 回测和性能分析
5. 结果可视化和报告生成
"""

import os
import sys
import logging
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime

# 先创建logs目录，确保日志文件可以正常写入
os.makedirs('logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_model_{datetime.now().strftime("%y%m%d_%H%M")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 修复Windows终端编码问题
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.encoding = 'utf-8'
logger = logging.getLogger('main')

# 导入自定义模块
from config import Config
from data_loader import FutureDataLoader
from feature_engineer import FeatureEngineer
from model_trainer import LightGBMTrainer
from backtester import TradingBacktester


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='生产级LightGBM期货交易模型')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR, 
                      help='数据目录路径')
    parser.add_argument('--symbol', type=str, default=None, 
                      help='特定交易品种代码，不指定则使用所有品种')
    
    # 模型相关参数
    parser.add_argument('--model-version', type=str, default=Config.MODEL_VERSION, 
                      help='模型版本号')
    parser.add_argument('--optimize', action='store_true', default=True, 
                      help='是否进行超参数优化')
    parser.add_argument('--n-trials', type=int, default=Config.OPTUNA_N_TRIALS, 
                      help='Optuna优化的试验次数')
    
    # 回测相关参数
    parser.add_argument('--backtest', action='store_true', default=True, 
                      help='是否进行回测')
    parser.add_argument('--signal-threshold', type=float, default=0.0, 
                      help='信号阈值')
    
    # 其他参数
    parser.add_argument('--save-results', action='store_true', default=True, 
                      help='是否保存结果')
    parser.add_argument('--plot-results', action='store_true', default=True, 
                      help='是否绘制结果图表')
    
    return parser.parse_args()


def setup_directories():
    """
    设置工作目录
    """
    # 创建必要的目录
    directories = [
        Config.DATA_DIR,
        Config.PROCESSED_DATA_DIR,
        Config.MODELS_DIR,
        Config.PLOTS_DIR,
        Config.BACKTEST_DIR,
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"目录已确保存在: {directory}")


def main():
    """
    主函数
    """
    start_time = time.time()
    logger.info("===== 开始执行LightGBM期货交易模型 =====")
    
    # 解析命令行参数
    args = parse_arguments()
    logger.info(f"命令行参数: {args}")
    
    # 设置目录
    setup_directories()
    
    # 1. 数据加载和预处理
    logger.info("\n===== 数据加载和预处理 =====")
    # 使用data_loader的默认文件夹加载训练数据
    data_loader = FutureDataLoader()
    data_files = data_loader.find_data_files()
    
    if not data_files:
        logger.error("没有找到数据文件，请检查数据目录")
        return
    
    logger.info(f"找到 {len(data_files)} 个数据文件")
    for file_path in data_files:
        logger.info(f"找到数据文件: {file_path}")
    
    # 加载和预处理所有文件
    df_combined = pd.DataFrame()
    for file_path in data_files:
        # 主程序执行
        try:
            logger.info(f"处理文件: {file_path}")
            
            # 加载单个文件
            df = data_loader.load_single_file(file_path)
            
            # 清洗数据
            df = data_loader.clean_data(df)
            
            # 生成目标变量
            df = data_loader.generate_target(df)
            
            # 添加到组合数据中
            if df_combined.empty:
                df_combined = df
            else:
                # 检查是否有重叠的时间戳
                df_combined = pd.concat([df_combined, df[~df.index.isin(df_combined.index)]])
                
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {str(e)}")
            continue
    
    if df_combined.empty:
        logger.error("所有数据文件处理失败，请检查数据质量")
        return
    
    # 确保时间索引排序 - 彻底解决时区混合问题
    try:
        # 强制将索引转换为字符串，然后排序
        df_combined['temp_index'] = df_combined.index.astype(str)
        df_combined.sort_values('temp_index', inplace=True)
        df_combined.drop('temp_index', axis=1, inplace=True)
        logger.info(f"数据处理完成，总样本数: {len(df_combined)}")
    except Exception as e:
        logger.error(f"索引排序失败: {str(e)}")
        # 作为最后的手段，使用reset_index和直接排序
        df_combined.reset_index(inplace=True)
        df_combined.sort_values('index', inplace=True)
        df_combined.set_index('index', inplace=True)
        logger.info(f"使用reset_index后排序完成，总样本数: {len(df_combined)}")
    
    # 3. 数据分割
    logger.info("\n===== 数据分割 =====")
    # 先分割数据
    train_df_raw, val_df_raw, test_df_raw = data_loader.split_data(df_combined)
    
    # 2. 特征工程（现在在分割后的数据上进行）
    logger.info("\n===== 特征工程 =====")
    feature_engineer = FeatureEngineer()
    
    # 在训练集上执行特征工程
    logger.info("在训练集上执行特征工程...")
    train_df, feature_names, _ = feature_engineer.engineer_all_features(train_df_raw)
    
    # 在验证集上执行特征工程
    logger.info("在验证集上执行特征工程...")
    val_df, _, _ = feature_engineer.engineer_all_features(val_df_raw)
    
    # 在测试集上执行特征工程
    logger.info("在测试集上执行特征工程...")
    test_df, _, _ = feature_engineer.engineer_all_features(test_df_raw)
    
    # 检查特征数量
    logger.info(f"生成了 {len(feature_names)} 个特征")
    logger.info(f"前10个特征: {feature_names[:10]}")
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    
    # 4. 模型训练
    logger.info("\n===== 模型训练 =====")
    trainer = LightGBMTrainer()
    
    # 准备训练数据
    X_train, y_train, valid_features = trainer.prepare_data_for_training(
        train_df, feature_names
    )
    X_val, y_val, _ = trainer.prepare_data_for_training(
        val_df, valid_features
    )
    X_test, y_test, _ = trainer.prepare_data_for_training(
        test_df, valid_features
    )
    
    # 调整Optuna参数
    original_trials = Config.OPTUNA_N_TRIALS
    Config.OPTUNA_N_TRIALS = args.n_trials
    
    # 训练模型
    model = trainer.train_model(
        X_train, y_train, X_val, y_val, 
        valid_features, 
        optimize=args.optimize
    )
    
    # 恢复原始配置
    Config.OPTUNA_N_TRIALS = original_trials
    
    # 5. 模型评估
    logger.info("\n===== 模型评估 =====")
    eval_results = trainer.evaluate_model(X_test, y_test)
    
    # 6. 时间序列交叉验证（可选）
    run_cv = getattr(Config, 'RUN_CROSS_VALIDATION', False)
    if run_cv:
        logger.info("\n===== 时间序列交叉验证 =====")
        cv_n_splits = getattr(Config, 'CV_N_SPLITS', 5)
        cv_results = trainer.time_series_cross_validation(
            np.vstack([X_train, X_val, X_test]),
            np.concatenate([y_train, y_val, y_test]),
            valid_features,
            n_splits=cv_n_splits
        )
    
    # 7. 绘制训练结果
    if args.plot_results:
        logger.info("\n===== 绘制训练结果图表 =====")
        trainer.plot_training_results()
    
    # 8. 保存模型
    if args.save_results:
        logger.info("\n===== 保存模型 =====")
        model_name = trainer.save_model()
        logger.info(f"模型已保存为: {model_name}")
    
    # 9. 回测
    if args.backtest:
        logger.info("\n===== 交易回测 =====")
        backtester = TradingBacktester()
        
        # 在测试集上预测
        y_pred_class, y_pred_proba = trainer.predict(X_test)
        
        # 运行回测
        backtest_results = backtester.run_backtest(
            test_df, y_pred_class, y_pred_proba, 
            signal_threshold=args.signal_threshold
        )
        
        # 计算性能指标
        performance_metrics = backtester.calculate_performance_metrics(backtest_results)
        
        # 市场状态分析
        market_state_analysis = backtester.analyze_by_market_state(backtest_results)
        
        # 绘制回测结果
        if args.plot_results:
            logger.info("绘制回测结果图表")
            backtester.plot_backtest_results(backtest_results, performance_metrics)
        
        # 敏感性分析
        run_sensitivity = getattr(Config, 'RUN_SENSITIVITY_ANALYSIS', False)
        if run_sensitivity:
            logger.info("进行参数敏感性分析")
            sensitivity_df = backtester.sensitivity_analysis(
                test_df, y_pred_class, y_pred_proba
            )
        
        # 生成交易建议
        logger.info("\n===== 交易策略建议 =====")
        recommendations = backtester.get_trading_recommendations(performance_metrics)
        for rec in recommendations:
            logger.info(f"[{rec['type']}] {rec['message']}")
        
        # 生成性能报告
        if args.save_results:
            logger.info("生成性能报告")
            report_path = backtester.generate_performance_report(
                performance_metrics, market_state_analysis
            )
            logger.info(f"性能报告已保存至: {report_path}")
            
            # 保存回测结果
            backtester.save_backtest_results(backtest_results, performance_metrics)
    
    # 10. 总结
    total_time = time.time() - start_time
    logger.info(f"\n===== 执行完成 =====")
    logger.info(f"总耗时: {total_time:.2f} 秒")
    
    if args.backtest and 'performance_metrics' in locals():
        basic_metrics = performance_metrics['basic']
        logger.info(f"\n最终性能指标:")
        logger.info(f"总收益率: {basic_metrics['total_return_pct']:.2f}%")
        logger.info(f"年化收益率: {basic_metrics['annual_return_pct']:.2f}%")
        logger.info(f"夏普比率: {basic_metrics['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {basic_metrics['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)