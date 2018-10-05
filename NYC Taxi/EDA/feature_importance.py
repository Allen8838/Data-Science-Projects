import pandas as pd

def find_feature_imp(model, features_used):
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(features_used))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                    'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': features_used})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)

    return feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)