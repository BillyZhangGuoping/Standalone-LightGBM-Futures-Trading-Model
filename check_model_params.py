import lightgbm as lgb
import numpy as np

# 加载模型
model = lgb.Booster(model_file='models_lgbm/lgbm_model_v1.0_251106_1643.txt')

# 打印模型参数
print('模型参数:')
for key, value in model.params.items():
    print(f'  {key}: {value}')

# 查看树的数量
print(f'\n树的数量: {model.num_trees()}')

# 查看特征重要性
print('\n特征重要性:')
importance = model.feature_importance(importance_type='split')
feature_names = model.feature_name()
for i in range(len(importance)):
    if importance[i] > 0:
        print(f'  {feature_names[i]}: {importance[i]}')