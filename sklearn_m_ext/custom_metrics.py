import warnings
from pprint import pprint
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics._classification as clf_metrics


def _calc_csi(tp: int, fp: int, fn: int) -> float:
    """
    Calculate critical success index defined as. Can be interpreted as accuracy that considers rare events, the class
    that is not rare should be the false class. Values range from [0, 1] where 1 is perfect accuracy?

    :param tp: true positives
    :param fp: false positives
    :param fn: false negatives

    :return:critical success index
    """
    return tp / (tp + fp + fn)


def _calc_bs(tp: int, fp: int, fn: int) -> float:
    """
    Calculate the brier score

    :param tp:
    :param fp:
    :param fn:

    :return:
    """
    return (tp + fp) / (tp + fn)


def _calc_hit_rate(tp, fn):
    return tp / (tp + fn)


def _calc_false_alarm_rate(fp, tn):
    return fp / (fp + tn)


def _calc_pss(tp, fp, tn, fn):
    """
    Peirce Skill Score

    :param tp:
    :param fp:
    :param tn:
    :param fn:

    :return:
    """
    return _calc_hit_rate(tp, fn) - _calc_false_alarm_rate(fp, tn)


def _calc_or(tp, fp, tn, fn):
    """
    Odds ratio

    :param tp:
    :param fp:
    :param tn:
    :param fn:

    :return:
    """
    H = _calc_hit_rate(tp, fn)
    F = _calc_false_alarm_rate(fp, tn)
    return (H / (1 - H)) / (F / (1 - F))


def _calc_orss(tp, fp, tn, fn):
    """
    Odds Ratio Skill score

    :param tp:
    :param fp:
    :param tn:
    :param fn:

    :return:
    """
    OR = _calc_or(tp, fp, tn, fn)
    return (OR - 1) / (OR + 1)


def _csi_from_cm(cm):
    """
    Calculate the cristical sucess index from a binary confusion matrix. The confusion matrix must be defined as
    [[TN, FN], [FP, TP]], i.e a the reverse of what is shown at wikipedia https://en.wikipedia.org/wiki/Confusion_matrix
    :param cm:
    :return:
    """
    tp, fp, _, fn = _extract_values_from_bin_contingency_matrix(cm)
    return _calc_csi(tp, fp, fn)


def _bs_from_cm(cm):
    tp, fp, _, fn = _extract_values_from_bin_contingency_matrix(cm)
    return _calc_bs(tp, fp, fn)


def _pss_from_cm(cm):
    tp, fp, tn, fn = _extract_values_from_bin_contingency_matrix(cm)
    return _calc_pss(tp, fp, tn, fn)


def _or_from_cm(cm):
    tp, fp, tn, fn = _extract_values_from_bin_contingency_matrix(cm)
    return _calc_or(tp, fp, tn, fn)


def _orss_from_cm(cm):
    tp, fp, tn, fn = _extract_values_from_bin_contingency_matrix(cm)
    return _calc_orss(tp, fp, tn, fn)


def _extract_values_from_bin_contingency_matrix(cm):
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    return tp, fp, tn, fn


def _calc_score(function: str, y_true, y_pred, sample_weight=None, force_multilabel=False):
    """
    Implement all scores above
    :param function: Name of the function, mapped in _FUNCTIONS
    :param y_true:
    :param y_pred:
    :param normalize:
    :param sample_weights:
    :return:
    """
    _FUNCTIONS = {"brier_score": _bs_from_cm, "critical_success_index": _csi_from_cm, "peirce_skill_score": _pss_from_cm,
                  "odds_ratio": _or_from_cm, "odds_ratio_skill_score": _orss_from_cm}
    y_type, y_true, y_pred = clf_metrics._check_targets(y_true, y_pred)
    if force_multilabel:
        labels = np.unique(y_true)
        cm = clf_metrics.multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, samplewise=None,
                                                     labels=labels)
        val = {labels[i]: _FUNCTIONS[function](cm[i]) for i in range(len(cm))}
    else:
        if y_type == "binary":
            cm = confusion_matrix(y_true, y_pred)
            val = _FUNCTIONS[function](cm)

        elif y_type.startswith("multiclass"):
            labels = np.unique(y_true)
            cm = clf_metrics.multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, samplewise=None, labels=labels)
            val = {labels[i]: _FUNCTIONS[function](cm[i]) for i in range(len(cm))}

        else:
            val = np.nan
            warnings.warn("%s could no be calculated undefined y_type %s" %(function, y_type))

    return val


def csi_score(y_true, y_pred, sample_weight=None):
    return _calc_score("critical_success_index", y_true, y_pred, sample_weight=sample_weight)


def bs_score(y_true, y_pred, sample_weight=None):
    return _calc_score("brier_score", y_true, y_pred, sample_weight=sample_weight)


def pss_score(y_true, y_pred, sample_weight=None):
    return _calc_score("peirce_skill_score", y_true, y_pred, sample_weight=sample_weight)


def or_score(y_true, y_pred, sample_weight=None):
    return _calc_score("odds_ratio", y_true, y_pred, sample_weight=sample_weight)


def orss_score(y_true, y_pred, sample_weight=None):
    return _calc_score("odds_ratio_skill_score", y_true, y_pred, sample_weight=sample_weight)

def classification_report2(y_true, y_pred, sample_weight=None, digits=4, return_dict=False):
    """
    Consisting of ORSS, PSS and BS, because accoding to Stephenson, D.B., 2000, “Use of the ‘Odds Ratio’ for Diagnosing Forecast Skill,” Weather and
    Forecasting 15(2):221–232. (From paper in Readme)
    they can describe a 2x2 contigency table.

    :param y_true:
    :param y_pred:
    :param sample_weight:
    :return:
    """
    _ADD_SUPPORT = True
    res = {}
    for function in ("brier_score", "peirce_skill_score", "odds_ratio_skill_score"):
        res[function] = _calc_score(function, y_true, y_pred, sample_weight=sample_weight, force_multilabel=True)

    res_new = {key: {} for key in res.get("brier_score").keys()}
    _, counts = np.unique(y_true, return_counts=True)
    supports = list(zip(sorted(res_new.keys()), counts))
    for key in res.keys():
        for key_rev in res_new.keys():
            res_new[key_rev][key] = res[key][key_rev]

    if _ADD_SUPPORT:
        for tuple_ in supports:
            res_new[tuple_[0]]["support"] = tuple_[1]

    report ="\t" + "\t".join(list(res.keys()) + ["support"]) + "\n"
    for row in res_new.keys():
        report += str(row) + "\t"
        report += "\t".join([str(round(val, digits)) for val in res_new[row].values()]) + "\n"

    #pprint(res_new)
    result = report
    if return_dict:
        result = res_new
    return result


