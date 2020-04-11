import pickle
import pandas as pd
import settings
from sklearn.preprocessing import MinMaxScaler

from utils.utils import get_logger, time_format


def main():
    logger = get_logger('make_prediction')

    with open('{}/baseline_xgb.pcl'.format(settings.MODEL_PATH), 'rb') as f:
        model = pickle.load(f)

    X = pd.read_csv('{}dataset_test.csv'.format(settings.DATASET_PATH), sep=';')
    X_mm = X.drop(['user_id'], axis=1)
    X_mm = MinMaxScaler().fit_transform(X_mm)

    predict_test = model.predict(X_mm)
    X_pred = pd.DataFrame(X['user_id'])
    X_pred['is_churned'] = predict_test

    X_pred.to_csv('{}dataset_pred.csv'.format(settings.PREDICTION_PATH), sep=';', index=False)

    logger.info('Prediction is successfully saved to {}'.format(settings.PREDICTION_PATH))


if __name__ == "__main__":
    main()
