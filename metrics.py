from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


def all_metric(labels,lab_predict):
    micro_auc = 0 
    macro_auc = 0 
    micro_f1 = 0 
    macro_f1 = 0 
    micro_precision = 0 
    macro_precision = 0 
    micro_recall = 0 
    macro_recall = 0 
    try:
        micro_auc = metrics.roc_auc_score(labels,lab_predict,average="micro")
    except:pass

    try:
        macro_auc = metrics.roc_auc_score(labels,lab_predict,average="macro")
    except:pass
    lab_predict=(lab_predict > 0.5) 
    try:

        micro_f1 = metrics.f1_score(labels,lab_predict,average="micro")
    except:pass

    try:
        macro_f1 = metrics.f1_score(labels,lab_predict,average="macro")
    except:pass
    try:

        micro_precision = metrics.precision_score(labels,lab_predict,average="micro")
    except:pass

    try:
        macro_precision = metrics.precision_score(labels,lab_predict,average="macro")
    except:pass
    try:

        micro_recall = metrics.recall_score(labels,lab_predict,average="micro")
    except:pass
    
    try:
        macro_recall = metrics.recall_score(labels,lab_predict,average="macro")
    except:pass
    return  micro_auc,macro_auc,micro_f1,macro_f1,micro_precision,macro_precision,micro_recall,macro_recall
