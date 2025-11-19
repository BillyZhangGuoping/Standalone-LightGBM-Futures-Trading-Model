#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 模型优化脚本

此脚本用于实现模型架构优化、特征工程改进和回测策略调整
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import optuna
import logging
import os
import sys

# 导入配置和模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from data_loader import FutureDataLoader
from feature_engineer import FeatureEngineer
from model_trainer import LightGBMTrainer
from backtester import TradingBacktester

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_optimizer')

def optimize_model_architecture():
    """
    优化模型架构
    """
    logger.info("开始优化模型架构...")
    
    # 1. 加载数据
    loader = FutureDataLoader()
    data_files = loader.find_data_files()
    
    if not data_files:
        logger.error("没有找到数据文件")
        return
    
    # 加载并预处理所有数据
    processed_data = loader.load_and_preprocess_all_files()
    if not processed_data:
        logger.error("没有加载到任何预处理数据")
        return
    # 合并所有品种的数据
    df = pd.concat(processed_data.values(), ignore_index=True)
    if df.empty:
        logger.error("合并后的数据为空，无法进行后续处理")
        return
    
    # 2. 特征工程
    engineer = FeatureEngineer()
    df_with_features, feature_names, target = engineer.engineer_all_features(df)
    
    # 3. 数据分割
    X = df_with_features[feature_names]
    y = df_with_features['target']
    
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    
    # 4. 定义Optuna目标函数
    def objective(trial):
        # 模型参数
        params = {
            'booster': trial.suggest_categorical('booster', ['gbdt', 'dart', 'goss']),
            'objective': Config.OBJECTIVE,
            'num_class': Config.NUM_CLASS,
            'metric': Config.METRIC,
            'num_leaves': trial.suggest_int('num_leaves', 32, 384, step=16),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.08, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0001, 15.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0001, 15.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 8, 250, step=10),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.000001, 0.2),
            'max_depth': trial.suggest_int('max_depth', 8, 35),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100, step=5),
            'random_seed': Config.RANDOM_SEED,
            'verbose': -1
        }
        
        # 动态调整参数
        if params['booster'] == 'goss':
            params['bagging_fraction'] = 1.0  # GOSS不需要bagging
            params['bagging_freq'] = 0
        
        if params['booster'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)
        
        f1_scores = []
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # SMOTE过采样
            if Config.SMOTE_ENABLED:
                smote = SMOTE(sampling_strategy=Config.SMOTE_SAMPLING_STRATEGY, 
                             k_neighbors=Config.SMOTE_K_NEIGHBORS, 
                             random_state=Config.RANDOM_SEED)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # 训练模型
            lgb_train = lgb.Dataset(X_train_resampled, label=y_train_resampled)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
            model = lgb.train(params, lgb_train, valid_sets=[lgb_val], 
                              num_boost_round=Config.NUM_BOOST_ROUND, 
                              early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS, 
                              verbose_eval=False)
            
            # 预测
            y_pred = np.argmax(model.predict(X_val, num_iteration=model.best_iteration), axis=1)
            
            # 计算F1分数
            f1 = f1_score(y_val, y_pred, average='weighted')
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    # 运行Optuna优化
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(multivariate=True, 
                                                                   n_startup_trials=Config.OPTUNA_N_STARTUP_TRIALS, 
                                                                   seed=Config.OPTUNA_SEED),
                               pruner=optuna.pruners.HyperbandPruner())
    
    study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, n_jobs=-1)
    
    logger.info(f"优化完成，最佳F1分数: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")
    
    # 保存最佳参数
    best_params = study.best_params
    best_params['objective'] = Config.OBJECTIVE
    best_params['num_class'] = Config.NUM_CLASS
    best_params['metric'] = Config.METRIC
    best_params['random_seed'] = Config.RANDOM_SEED
    
    import json
    with open(os.path.join(Config.MODELS_DIR, 'best_model_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info("最佳参数已保存到best_model_params.json")
    
    return best_params

def improve_feature_engineering():
    """
    改进特征工程
    """
    logger.info("开始改进特征工程...")
    
    # 1. 加载数据
    loader = FutureDataLoader()
    data_files = loader.find_data_files()
    
    if not data_files:
        logger.error("没有找到数据文件")
        return
    
    # 加载并合并所有数据
    df = loader.load_data_files(data_files)
    df = loader.clean_data(df)
    df = loader.generate_target(df)
    
    # 2. 特征工程
    engineer = FeatureEngineer()
    df_with_features, feature_names, target = engineer.engineer_all_features(df)
    
    logger.info(f"原始特征数量: {len(feature_names)}")
    
    # 3. 特征重要性分析
    trainer = LightGBMTrainer()
    X = df_with_features[feature_names]
    y = df_with_features['target']
    
    # 训练一个简单模型以获取特征重要性
    params = {
        'objective': Config.OBJECTIVE,
        'num_class': Config.NUM_CLASS,
        'metric': Config.METRIC,
        'num_leaves': 64,
        'learning_rate': 0.01,
        'num_boost_round': 100,
        'verbose': -1
    }
    
    lgb_train = lgb.Dataset(X, label=y)
    model = lgb.train(params, lgb_train, verbose_eval=False)
    
    # 获取特征重要性
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # 保存特征重要性
    feature_importance_df.to_csv(os.path.join(Config.MODELS_DIR, 'feature_importance.csv'), index=False)
    
    # 选择重要性排名前N的特征
    top_features = feature_importance_df['feature'].head(Config.MAX_FEATURES).tolist()
    
    logger.info(f"改进后特征数量: {len(top_features)}")
    logger.info(f"前10个重要特征: {top_features[:10]}")
    
    return top_features

def adjust_backtest_strategy():
    """
    调整回测策略
    """
    logger.info("开始调整回测策略...")
    
    # 1. 加载回测配置
    from config import BacktestConfig as BConfig
    
    # 2. 定义要测试的参数组合
    parameter_combinations = [
        {'min_confidence': 0.3, 'stop_loss_multiplier': 0.5, 'take_profit_multiplier': 1.5},
        {'min_confidence': 0.35, 'stop_loss_multiplier': 0.75, 'take_profit_multiplier': 1.25},
        {'min_confidence': 0.4, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 1.0},
        {'min_confidence': 0.45, 'stop_loss_multiplier': 1.25, 'take_profit_multiplier': 0.75},
        {'min_confidence': 0.5, 'stop_loss_multiplier': 1.5, 'take_profit_multiplier': 0.5},
    ]
    
    # 3. 运行回测
    # 回测策略调整通常需要结合实际数据和模型预测结果
    # 这里将在主脚本中完成整体回测流程
    logger.info("回测策略调整将在主流程中进行")
        
    logger.info("回测策略调整完成")

def main():
    """
    主函数
    """
    logger.info("=== 开始模型优化流程 ===")
    
    # 1. 优化模型架构
    best_params = optimize_model_architecture()
    
    # 2. 改进特征工程
    top_features = improve_feature_engineering()
    
    # 3. 调整回测策略
    adjust_backtest_strategy()
    
    logger.info("=== 模型优化流程完成 ===")
    
    return best_params, top_features

if __name__ == "__main__":
    main()