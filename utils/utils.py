import logging
from datetime import timedelta

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc, \
                            log_loss, roc_auc_score, average_precision_score, confusion_matrix


def time_format(sec):
    return str(timedelta(seconds=sec))


def get_logger(logger_nm: str):
    logger = logging.getLogger(logger_nm)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("../logs/ml_prod.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def evaluation(y_true, y_pred, y_prob, logger_nm):
    logger = get_logger(logger_nm)

    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)

    logger.info('Precision: {}'.format(precision))
    logger.info('Recall: {}'.format(recall))
    logger.info('F1: {}'.format(f1))
    logger.info('Log Loss: {}'.format(ll))
    logger.info('ROC AUC: {}'.format(roc_auc))

    return precision, recall, f1, ll, roc_auc

