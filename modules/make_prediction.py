import pickle
import pandas as pd
import settings
from sklearn.preprocessing import MinMaxScaler


def main():
    with open('{}/baseline_xgb.pcl'.format(settings.MODEL_PATH), 'rb') as f:
        model = pickle.load(f)

    X = pd.read_csv('{}dataset_test.csv'.format(settings.DATASET_PATH), sep=';')
    #X_mm = MinMaxScaler().fit_transform(X)

    predict_test = model.predict(X)
    print(predict_test.head())


if __name__ == "__main__":
    main()