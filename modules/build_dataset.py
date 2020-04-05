import time
from datetime import datetime, timedelta
import pandas as pd
import settings
from utils.utils import time_format, get_logger
from argparse import ArgumentParser


def build_dataset_raw(inter_list: list,
                      raw_data_path: str,
                      dataset_path: str,
                      mode: str):
    logger = get_logger('build_dataset')

    start_t = time.time()

    logger.info('Dataset type: {}'.format(mode))
    logger.info('Start reading csv files: {}'.format(time_format(time.time() - start_t)))

    sample = pd.read_csv('{}{}/sample.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    profiles = pd.read_csv('{}{}/profiles.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    payments = pd.read_csv('{}{}/payments.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    reports = pd.read_csv('{}{}/reports.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    abusers = pd.read_csv('{}{}/abusers.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    logins = pd.read_csv('{}{}/logins.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    pings = pd.read_csv('{}{}/pings.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    sessions = pd.read_csv('{}{}/sessions.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    shop = pd.read_csv('{}{}/shop.csv'.format(raw_data_path,mode), sep=';', na_values=['\\N', 'None'], encoding='utf-8')

    logger.info('Run time (reading csv files): {}'.format(time_format(time.time() - start_t)))
    # -----------------------------------------------------------------------------------------------------
    logger.info('NO dealing with outliers, missing values and categorical features...')
    # -----------------------------------------------------------------------------------------------------
    # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

    logger.info('Creating dataset...')
    # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
    if mode == 'train':
        dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
    elif mode == 'test':
        dataset = sample.copy()[['user_id', 'level', 'donate_total']]

    # Пройдемся по всем источникам, содержащим "динамичекие" данные
    for df in [payments, reports, abusers, logins, pings, sessions, shop]:

        # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
        data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
        data['day_num_before_churn'] = 1 + (data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                                            data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(
            lambda x: x.days)
        df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

        # Для каждого признака создадим признаки для каждого из времененно интервала (в нашем примере 4 интервала по
        # 7 дней)
        features = list(set(data.columns) - {'user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn'})
        logger.info('Processing with features:{}'.format(features))
        for feature in features:
            for i, inter in enumerate(inter_list):
                inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)]. \
                    groupby('user_id')[feature].mean().reset_index(). \
                    rename(index=str, columns={feature: feature + '_{}'.format(i + 1)})
                df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

        # Добавляем построенные признаки в датасет
        dataset = pd.merge(dataset, df_features, how='left', on='user_id')

        logger.info('Run time (calculating features): {}'.format(time_format(time.time() - start_t)))

    # Добавляем "статические" признаки
    dataset = pd.merge(dataset, profiles, on='user_id')
    # ------------------------------------------------------------------------------------------------------------------
    dataset.to_csv('{}dataset_raw_{}.csv'.format(dataset_path, mode), sep=';', index=False)
    print('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
          format(dataset_path, time_format(time.time() - start_t)))
    logger.info('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
                format(dataset_path, time_format(time.time() - start_t)))


def prepare_dataset(inter_list: list,
                    dataset_type: str,
                    dataset_path: str):
    logger = get_logger('prepare_dataset')

    logger.info(dataset_type)
    start_t = time.time()

    dataset = pd.read_csv('{}/dataset_raw_{}.csv'.format(dataset_path, dataset_type), sep=';')

    logger.info('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1, len(inter_list) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)

    logger.info('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
          format(dataset_path, time_format(time.time() - start_t)))


def main():
    # Разбираем аргументы командной строки
    parser = ArgumentParser()

    parser.add_argument(
        '-m', '--mode', type=str, required=False, help=''
    )

    args = parser.parse_args()

    if args.mode == 'train' or args.mode == 'all' or args.mode is None:
        build_dataset_raw(inter_list=settings.INTER_LIST,
                          raw_data_path=settings.RAW_DATA_PATH,
                          dataset_path=settings.DATASET_PATH,
                          mode='train')
        prepare_dataset(inter_list=settings.INTER_LIST,
                        dataset_type='train',
                        dataset_path=settings.DATASET_PATH)

    if args.mode == 'test' or args.mode == 'all' or args.mode is None:
        build_dataset_raw(inter_list=settings.INTER_LIST,
                          raw_data_path=settings.RAW_DATA_PATH,
                          dataset_path=settings.DATASET_PATH,
                          mode='test')
        prepare_dataset(inter_list=settings.INTER_LIST,
                        dataset_type='test',
                        dataset_path=settings.DATASET_PATH)



if __name__ == "__main__":
    main()
