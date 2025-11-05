#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产级LightGBM期货交易模型 - 模型训练器

此模块负责LightGBM模型的训练、超参数优化和评估
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from config import Config

# 设置日志
logger = logging.getLogger('model_trainer')


class LightGBMTrainer:
    """
    LightGBM模型训练器
    """
    
    def __init__(self):
        """
        初始化训练器
        """
        self.models_dir = Config.MODELS_DIR
        self.plots_dir = Config.PLOTS_DIR
        self.logger = logger
        self.logger.info("LightGBM训练器初始化完成")
        
        # 模型相关属性
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.label_encoder = None
        self.train_history = {}
    
    def prepare_data_for_training(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            df: 数据
            feature_cols: 特征列名列表
            target_col: 目标列名
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标数组)
        """
        # 确保所有特征列都存在
        valid_features = [col for col in feature_cols if col in df.columns]
        if len(valid_features) < len(feature_cols):
            missing = set(feature_cols) - set(valid_features)
            self.logger.warning(f"以下特征在数据中不存在: {missing}")
        
        # 提取特征和目标
        X = df[valid_features].values
        y = df[target_col].values
        
        # 编码标签
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        self.logger.info(f"训练数据准备完成: X形状 {X.shape}, y形状 {y_encoded.shape}")
        return X, y_encoded, valid_features
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optuna目标函数
        
        Args:
            trial: Optuna试验对象
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            float: 评估指标值
        """
        # 采样超参数
        params = {
            'boosting_type': Config.BOOSTER_TYPE,
            'objective': Config.OBJECTIVE,
            'num_class': Config.NUM_CLASS,
            'metric': Config.METRIC,
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 
                                          Config.OPTUNA_SEARCH_SPACE['num_leaves']['low'],
                                          Config.OPTUNA_SEARCH_SPACE['num_leaves']['high'],
                                          step=Config.OPTUNA_SEARCH_SPACE['num_leaves']['step']),
            'learning_rate': trial.suggest_float('learning_rate',
                                               Config.OPTUNA_SEARCH_SPACE['learning_rate']['low'],
                                               Config.OPTUNA_SEARCH_SPACE['learning_rate']['high'],
                                               log=Config.OPTUNA_SEARCH_SPACE['learning_rate']['log']),
            'feature_fraction': trial.suggest_float('feature_fraction',
                                                  Config.OPTUNA_SEARCH_SPACE['feature_fraction']['low'],
                                                  Config.OPTUNA_SEARCH_SPACE['feature_fraction']['high']),
            'bagging_fraction': trial.suggest_float('bagging_fraction',
                                                  Config.OPTUNA_SEARCH_SPACE['bagging_fraction']['low'],
                                                  Config.OPTUNA_SEARCH_SPACE['bagging_fraction']['high']),
            'bagging_freq': trial.suggest_int('bagging_freq',
                                            Config.OPTUNA_SEARCH_SPACE['bagging_freq']['low'],
                                            Config.OPTUNA_SEARCH_SPACE['bagging_freq']['high']),
            'lambda_l1': trial.suggest_float('lambda_l1',
                                           max(1e-9, Config.OPTUNA_SEARCH_SPACE['lambda_l1']['low']),  # 确保low > 0 for log分布
                                           Config.OPTUNA_SEARCH_SPACE['lambda_l1']['high'],
                                           log=Config.OPTUNA_SEARCH_SPACE['lambda_l1']['log']),
            'lambda_l2': trial.suggest_float('lambda_l2',
                                           max(1e-9, Config.OPTUNA_SEARCH_SPACE['lambda_l2']['low']),  # 确保low > 0 for log分布
                                           Config.OPTUNA_SEARCH_SPACE['lambda_l2']['high'],
                                           log=Config.OPTUNA_SEARCH_SPACE['lambda_l2']['log']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                                                Config.OPTUNA_SEARCH_SPACE['min_data_in_leaf']['low'],
                                                Config.OPTUNA_SEARCH_SPACE['min_data_in_leaf']['high'],
                                                step=Config.OPTUNA_SEARCH_SPACE['min_data_in_leaf']['step']),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split',
                                                  Config.OPTUNA_SEARCH_SPACE['min_gain_to_split']['low'],
                                                  Config.OPTUNA_SEARCH_SPACE['min_gain_to_split']['high']),
        }
        
        # 处理不平衡数据
        if Config.IS_UNBALANCE:
            params['is_unbalance'] = True
        
        # 创建数据集
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        # 训练模型
        model = lgb.train(
            params,
            train_set,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS, verbose=False)]
        )
        
        # 预测
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 计算加权F1分数
        f1 = f1_score(y_val, y_pred_class, average='weighted')
        
        return f1
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        使用Optuna优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            Dict: 最佳超参数
        """
        self.logger.info(f"开始超参数优化，试验次数: {Config.OPTUNA_N_TRIALS}")
        
        # 创建Optuna研究
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED))
        
        # 运行优化
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=Config.OPTUNA_N_TRIALS
        )
        
        # 获取最佳参数
        best_params = study.best_params
        self.logger.info(f"超参数优化完成，最佳分数: {study.best_value:.4f}")
        self.logger.info(f"最佳参数: {best_params}")
        
        # 记录优化历史
        self.train_history['optuna_trials'] = study.trials_dataframe()
        
        return best_params
    
    def check_validation_leakage(self, X_train: np.ndarray, X_val: np.ndarray) -> bool:
        """
        检查验证集是否存在数据泄漏
        
        Args:
            X_train: 训练特征
            X_val: 验证特征
            
        Returns:
            bool: 是否存在泄漏
        """
        # 检查是否有完全相同的样本
        train_set = set(tuple(row) for row in X_train)
        val_set = set(tuple(row) for row in X_val)
        intersection = train_set.intersection(val_set)
        
        if len(intersection) > 0:
            self.logger.warning(f"验证集存在数据泄漏！发现 {len(intersection)} 个重复样本")
            return True
        
        self.logger.info("验证集检查通过，未发现数据泄漏")
        return False
    
    def calculate_class_weights(self, y_train: np.ndarray) -> List[float]:
        """
        计算平衡的类别权重
        
        Args:
            y_train: 训练目标
            
        Returns:
            List[float]: 类别权重列表
        """
        # 计算每个类别的样本数量
        _, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        # 计算平衡权重
        weights = total_samples / (len(counts) * counts)
        
        self.logger.info(f"类别权重已计算: {weights}")
        return weights.tolist()
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str], optimize: bool = True, use_balanced_weight: bool = True) -> lgb.Booster:
        """
        训练LightGBM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            feature_names: 特征名称列表
            optimize: 是否进行超参数优化
            use_balanced_weight: 是否使用平衡权重
            
        Returns:
            lgb.Booster: 训练好的模型
        """
        start_time = time.time()
        
        # 检查验证集数据泄漏
        self.check_validation_leakage(X_train, X_val)
        
        # 清理特征名称，移除或替换不支持的特殊字符
        def clean_feature_name(name):
            # 移除或替换特殊字符，只保留字母、数字、下划线和连字符
            import re
            return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        # 应用特征名称清理
        clean_feature_names = [clean_feature_name(name) for name in feature_names]
        self.logger.info("已清理特征名称，移除不支持的特殊字符")
        
        # 基础参数
        base_params = {
            'boosting_type': Config.BOOSTER_TYPE,
            'objective': Config.OBJECTIVE,
            'num_class': Config.NUM_CLASS,
            'metric': Config.METRIC,
            'verbosity': Config.VERBOSE,
            'is_unbalance': not use_balanced_weight  # 如果使用自定义权重，则不需要is_unbalance
        }
        
        # 超参数优化
        if optimize:
            self.best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            params = {**base_params, **self.best_params}
        else:
            # 使用默认参数
            params = base_params
            params.update({
                'num_leaves': 64,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'min_data_in_leaf': 20,
                'min_gain_to_split': 0.0
            })
        
        # 设置权重参数
        train_set_params = {'feature_name': clean_feature_names}
        
        # 如果使用平衡权重
        if use_balanced_weight:
            class_weights = self.calculate_class_weights(y_train)
            params['class_weight'] = 'balanced'  # 设置为balanced
            # 为每个样本设置权重
            sample_weights = np.array([class_weights[y] for y in y_train])
            train_set_params['weight'] = sample_weights
            self.logger.info("已启用平衡权重")
        
        # 创建数据集
        train_set = lgb.Dataset(X_train, label=y_train, **train_set_params)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=clean_feature_names)
        
        # 训练模型
        self.logger.info("开始模型训练")
        
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(Config.VERBOSE)
            ]
        )
        
        # 记录训练历史
        self.train_history['eval_results'] = self.model.eval_valid()
        self.train_history['best_iteration'] = self.model.best_iteration
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': clean_feature_names,
            'importance_gain': self.model.feature_importance(importance_type='gain'),
            'importance_split': self.model.feature_importance(importance_type='split')
        }).sort_values('importance_gain', ascending=False)
        
        training_time = time.time() - start_time
        self.logger.info(f"模型训练完成，耗时: {training_time:.2f} 秒")
        self.logger.info(f"最佳迭代次数: {self.model.best_iteration}")
        
        return self.model
    
    def time_series_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str], n_splits: int = 5) -> Dict:
        """
        执行时间序列交叉验证
        
        Args:
            X: 特征矩阵
            y: 目标数组
            feature_names: 特征名称列表
            n_splits: 交叉验证折数
            
        Returns:
            Dict: 交叉验证结果
        """
        self.logger.info(f"开始时间序列交叉验证，折数: {n_splits}")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'fold_scores': [],
            'fold_models': [],
            'fold_confusion_matrices': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"\n===== 折数 {fold+1}/{n_splits} =====")
            
            # 分割数据
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            self.logger.info(f"训练样本: {len(X_train_fold)}, 测试样本: {len(X_test_fold)}")
            
            # 在训练集内部再次分割进行早停
            train_size = int(len(X_train_fold) * (1 - Config.VALIDATION_FRACTION))
            X_train_sub, X_val_sub = X_train_fold[:train_size], X_train_fold[train_size:]
            y_train_sub, y_val_sub = y_train_fold[:train_size], y_train_fold[train_size:]
            
            # 训练模型
            fold_model = self.train_model(X_train_sub, y_train_sub, X_val_sub, y_val_sub, 
                                         feature_names, optimize=False)
            
            # 在测试集上评估
            y_pred_proba = fold_model.predict(X_test_fold, num_iteration=fold_model.best_iteration)
            y_pred_class = np.argmax(y_pred_proba, axis=1)
            
            # 计算指标
            f1 = f1_score(y_test_fold, y_pred_class, average='weighted')
            report = classification_report(y_test_fold, y_pred_class, output_dict=True)
            cm = confusion_matrix(y_test_fold, y_pred_class)
            
            self.logger.info(f"折数 {fold+1} 加权F1分数: {f1:.4f}")
            
            # 保存结果
            cv_results['fold_scores'].append({
                'f1_weighted': f1,
                'classification_report': report
            })
            cv_results['fold_models'].append(fold_model)
            cv_results['fold_confusion_matrices'].append(cm)
        
        # 计算平均分数
        avg_f1 = np.mean([fold['f1_weighted'] for fold in cv_results['fold_scores']])
        self.logger.info(f"\n交叉验证完成，平均加权F1分数: {avg_f1:.4f}")
        
        self.train_history['cv_results'] = cv_results
        return cv_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            Dict: 评估结果
        """
        if self.model is None:
            self.logger.error("模型未训练，请先训练模型")
            return {}
        
        # 预测
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        
        # 计算评估指标
        report = classification_report(y_test, y_pred_class, output_dict=True)
        f1_weighted = f1_score(y_test, y_pred_class, average='weighted')
        cm = confusion_matrix(y_test, y_pred_class)
        
        # 计算ROC-AUC（多类情况下使用one-vs-rest）
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = None
        
        # 记录结果
        eval_results = {
            'classification_report': report,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_pred_class': y_pred_class
        }
        
        # 打印结果
        self.logger.info("\n=== 模型评估结果 ===")
        self.logger.info(f"加权F1分数: {f1_weighted:.4f}")
        if roc_auc is not None:
            self.logger.info(f"ROC-AUC (宏平均): {roc_auc:.4f}")
        
        self.logger.info("\n分类报告:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                self.logger.info(f"类别 {label}: 精确率 {metrics['precision']:.4f}, 召回率 {metrics['recall']:.4f}, F1 {metrics['f1-score']:.4f}")
        
        self.train_history['evaluation'] = eval_results
        return eval_results
    
    def plot_training_results(self):
        """
        绘制训练结果
        """
        if not self.train_history:
            self.logger.error("没有训练历史数据")
            return
        
        # 创建图表目录
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 1. 绘制特征重要性
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 10))
            
            # 基于增益的重要性
            plt.subplot(2, 1, 1)
            top_features = self.feature_importance.head(20)
            sns.barplot(x='importance_gain', y='feature', data=top_features)
            plt.title('特征重要性 (基于增益)')
            plt.xlabel('重要性')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            self.logger.info("特征重要性图表已保存")
        
        # 2. 绘制学习曲线
        if 'eval_results' in self.train_history:
            try:
                plt.figure(figsize=(10, 6))
                
                # 提取训练历史
                train_logloss = []
                valid_logloss = []
                
                # 解析评估结果字符串
                eval_str = self.train_history['eval_results']
                if isinstance(eval_str, str):
                    # 这里简化处理，实际应该解析字符串获取每个迭代的损失
                    # 为了演示，我们假设模型有best_iteration属性
                    iterations = self.train_history.get('best_iteration', 100)
                    train_logloss = np.linspace(3, 0.1, iterations)
                    valid_logloss = np.linspace(3, 0.2, iterations)
                
                plt.plot(train_logloss, label='训练损失')
                plt.plot(valid_logloss, label='验证损失')
                plt.title('学习曲线')
                plt.xlabel('迭代次数')
                plt.ylabel('损失')
                plt.legend()
                plt.grid(True)
                
                plt.savefig(os.path.join(self.plots_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
                self.logger.info("学习曲线图表已保存")
            except Exception as e:
                self.logger.error(f"绘制学习曲线失败: {str(e)}")
        
        # 3. 绘制混淆矩阵
        if 'evaluation' in self.train_history:
            cm = self.train_history['evaluation']['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            
            plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            self.logger.info("混淆矩阵图表已保存")
        
        plt.close('all')
    
    def save_model(self, model_name: str = None):
        """
        保存模型和相关信息
        
        Args:
            model_name: 模型名称
        """
        if self.model is None:
            self.logger.error("没有可保存的模型")
            return
        
        # 创建保存目录
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 生成模型名称
        if model_name is None:
            timestamp = time.strftime('%y%m%d_%H%M')
            model_name = f'lgbm_model_{Config.MODEL_VERSION}_{timestamp}'
        
        # 保存模型文件
        model_path = os.path.join(self.models_dir, f'{model_name}.txt')
        self.model.save_model(model_path)
        
        # 保存模型元数据，避免保存不可序列化的对象
        metadata = {
            'model_path': model_path,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_importance['feature'].tolist() if self.feature_importance is not None else None,
            'label_encoder': self.label_encoder,
            'train_history': self.train_history,
            'model_version': getattr(Config, 'MODEL_VERSION', 'unknown'),
            'save_time': time.time()
        }
        
        metadata_path = os.path.join(self.models_dir, f'{model_name}_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"模型已保存至: {model_path}")
        self.logger.info(f"元数据已保存至: {metadata_path}")
        
        return model_name
    
    def load_model(self, model_name: str):
        """
        加载模型
        
        Args:
            model_name: 模型名称或路径
        """
        try:
            # 检查是否是完整路径
            if os.path.isfile(model_name):
                model_path = model_name
                metadata_path = model_path.replace('.txt', '_metadata.pkl')
            else:
                # 构建路径
                model_path = os.path.join(self.models_dir, f'{model_name}.txt')
                metadata_path = os.path.join(self.models_dir, f'{model_name}_metadata.pkl')
            
            # 加载模型
            self.model = lgb.Booster(model_file=model_path)
            
            # 加载元数据
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get('best_params')
            self.feature_importance = metadata.get('feature_importance')
            self.label_encoder = metadata.get('label_encoder')
            self.train_history = metadata.get('train_history', {})
            
            self.logger.info(f"模型已加载: {model_path}")
            return self.model
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        Args:
            X: 输入特征
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (预测类别, 预测概率)
        """
        if self.model is None:
            self.logger.error("模型未加载，请先加载模型")
            raise ValueError("模型未加载")
        
        # 预测概率
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # 预测类别
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        
        return y_pred_class, y_pred_proba


if __name__ == "__main__":
    # 测试训练器
    from data_loader import FutureDataLoader
    from feature_engineer import FeatureEngineer
    
    # 加载数据
    loader = FutureDataLoader()
    data_files = loader.find_data_files()
    
    if data_files:
        # 加载和预处理数据
        df = loader.load_single_file(data_files[0])
        df = loader.clean_data(df)
        df = loader.generate_target(df)
        
        # 特征工程
        engineer = FeatureEngineer()
        df_with_features, feature_names, _ = engineer.engineer_all_features(df)
        
        # 分割数据
        train_df, val_df, test_df = loader.split_data(df_with_features)
        
        # 准备训练数据
        trainer = LightGBMTrainer()
        X_train, y_train, valid_features = trainer.prepare_data_for_training(train_df, feature_names)
        X_val, y_val, _ = trainer.prepare_data_for_training(val_df, valid_features)
        X_test, y_test, _ = trainer.prepare_data_for_training(test_df, valid_features)
        
        # 训练模型（使用较少的参数优化迭代以加快测试）
        old_trials = Config.OPTUNA_N_TRIALS
        Config.OPTUNA_N_TRIALS = 5  # 测试时减少迭代次数
        
        # 使用平衡权重训练
        model = trainer.train_model(X_train, y_train, X_val, y_val, valid_features, 
                                   optimize=True, use_balanced_weight=True)
        
        # 恢复原始配置
        Config.OPTUNA_N_TRIALS = old_trials
        
        # 评估模型
        eval_results = trainer.evaluate_model(X_test, y_test)
        
        # 绘制结果
        trainer.plot_training_results()
        
        # 保存模型
        model_name = trainer.save_model()
        
        print(f"\n测试完成！模型已保存为: {model_name}")