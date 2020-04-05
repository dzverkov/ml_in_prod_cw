import pickle

import settings
from utils.utils import evaluation
import xgboost as xgb
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def xgb_fit_predict(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight=3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_test, predict_test, predict_proba_test[:, 1], 'train_model')
    return clf


def main():
    dataset = pd.read_csv('{}dataset_train.csv'.format(settings.DATASET_PATH), sep=';')
    X = dataset.drop(['user_id', 'is_churned'], axis=1)
    y = dataset['is_churned']

    X_mm = MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=100)

    # Снизим дизбаланс классов
    X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_sample(X_train, y_train)

    model = xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)

    with open('{}/baseline_xgb.pcl'.format(settings.MODEL_PATH), 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
