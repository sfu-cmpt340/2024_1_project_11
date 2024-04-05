import lightgbm as lgb

def train_model(X_train, y_train):
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'num_leaves': 121,
        'learning_rate': 0.018623105710769177,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.756777580360579,
        'max_depth': 8,
        'verbose': 0
    }

    lgb_model = lgb.LGBMClassifier(**params)
    lgb_model.fit(X_train, y_train)
    return lgb_model
