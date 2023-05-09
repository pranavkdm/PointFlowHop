import numpy as np 
import pickle
from xgboost import XGBClassifier



with open('D:\PhD Electrical Engineering\Media Communications Lab\Scene Flow\SceneFlow-main\lib\pointhop.pkl', 'rb') as f:
    params = pickle.load(f, encoding='latin')

print(params.keys())
print(params['Layer_1_0_pca_params']['kernel'].shape)
print(params['Layer_1_0_pca_params']['pca_mean'].shape)
# print(params['Layer_1_0_pca_params']['energy'].shape)
# print(params['Layer_1_0_pca_params']['num_node'])


# clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#               gamma=0, gpu_id=-1, importance_type=None,
#               interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#               max_depth=3, min_child_weight=1, 
#               monotone_constraints='()', n_estimators=100, n_jobs=16,
#               num_parallel_tree=1, predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=2, subsample=0.8,
#               tree_method='approx', validate_parameters=1, verbosity=None)


# import pickle
# file_name = "xgb_reg.pkl"

# # save
# pickle.dump(clf, open(file_name, "wb"))

# # load
# xgb_model_loaded = pickle.load(open(file_name, "rb"))
# print(xgb_model_loaded.get_params())

# from sklearn.ensemble import GradientBoostingRegressor

# reg = GradientBoostingRegressor(random_state=0)
# print(reg.get_params)



