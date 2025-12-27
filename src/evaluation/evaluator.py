import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

class ClassificationEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names

    def compute_all_metrics(self, y_true, y_pred, y_probs):
        """
        Computes a full suite of classification metrics: Accuracy, Precision, Recall, F1, MCC, and AUC.
        """
        # Overall Scalari
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Macro and Weighted Averages
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # Area Under the ROC Curve (One-vs-Rest)
        auc_ovr = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

        # Per-Class Analysis (FPR, TPR, Specificity)
        cm = confusion_matrix(y_true, y_pred)
        per_class = {}
        
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 # Sensitivity/Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # Fall-out
            
            per_class[name] = {
                "tpr": round(float(tpr), 4),
                "fpr": round(float(fpr), 4),
                "precision": round(float(tp / (tp + fp) if (tp + fp) > 0 else 0), 4),
                "f1": round(float(2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0), 4)
            }

        return {
            "summary": {
                "accuracy": round(float(acc), 4),
                "mcc": round(float(mcc), 4),
                "f1_macro": round(float(f1_macro), 4),
                "f1_weighted": round(float(f1_weighted), 4),
                "auc_roc": round(float(auc_ovr), 4)
            },
            "per_class": per_class,
            "confusion_matrix": cm.tolist()
        }

    @staticmethod
    def save_metrics(metrics, path):
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)