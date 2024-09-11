
from collections import OrderedDict
import csv

import numpy as np

import torch
import torch.nn as nn
from torchvision.models import densenet121

#----------------------------------------------------------

def remove_module_prefix(state_dict):
    """
    Args:
        state_dict (OrderedDict): The state_dict containing the keys with 'module.' prefix
    Returns:
        new_state_dict (OrderedDict): The state_dict with the 'module.' prefix removed from the keys

    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def initialize_model():
    """
    Returns:
        model : The initialized DenseNet121 model with a modified classifier.

    """
    model = densenet121()
    model.classifier = nn.Linear(in_features=1024, out_features=1, bias=True)
    return model

def initialize_trained_model(trained_model_path):
    """
    Load the weights into the model.
    Args:
        trained_model_path (str): The path to the trained model weights.
    Returns:
        model: The initialized model with loaded weights.

    """
    saved_state = torch.load(trained_model_path)
    model = initialize_model()
    model.load_state_dict(remove_module_prefix(saved_state['state_dict']))

    return model


def get_prevelance(labels):
    """
    
    Args:
        labels (np.array): 1D array of labels

    Returns:
        prevalence (float): prevalence of positive labels
    
    """
    unique, counts = np.unique(labels, return_counts=True)
    prevs = dict(zip(unique, counts))
    negatives = prevs[0]
    positives = prevs[1]
    prevalence = float(positives)/(float(positives) + float(negatives))
    print('    Prevalence at {0:.2f}: ({1:d} positives | {2:d} negatives)'.format(prevalence, positives, negatives))
    return prevalence

def gen_performance_metrics(labels, probs, outfn, step_size=0.01):
    """
    Calc performance stepping across pSEA value and write performance metrics to file.

    Args:
        labels (np.array): 1D array of labels
        probs (np.array): 1D array of probablities
        outfn (str): output file name
        step_size (float): step size for cutoffs
    Returns:
        prevalence (float): prevalence of positive labels
        outfn (str): output file name
        all_TPRs (list): list of TPRs
        all_FPRs (list): list of FPRs
        all_PPVs (list): list of PPVs
    
    """
    perf_metrics = {}
    prevalence = get_prevelance(labels)

    all_TPRs, all_FPRs, all_PPVs = [], [], []
    thresholds = np.arange(0., 1. + step_size, step_size)
    print('')
    print('Calculating performance metrics') 
    for step in thresholds:
        # print('    Threshold for inclusion={0:.2f}'.format(step))
        if step == 0.:
            TPR, TNR, PPV, FPR, FNR, FDR = 1, 0, prevalence, 1, 0, 1-prevalence
            perf_metrics[step] = (TPR, TNR, PPV, FPR, FNR, FDR)
        elif step == 1.:
            TPR, TNR, PPV, FPR, FNR, FDR = 0, 1, 1, 0, 1, 0
            perf_metrics[step] = (TPR, TNR, PPV, FPR, FNR, FDR)
        else:
            #TP, FP, TN, FN = calc_truth_table_counts(labels, probs, step)
            pred_class = probs > step
            TP, FP, TN, FN = compute_tp_fp_tn_fn(pred_class=pred_class, targets=labels)
            # print(f'    \tTP={TP}  FP={FP}  TN={TN}  FN={FN}')
            TPR, TNR, PPV, FPR, FNR, FDR = calc_performance_metrics(TP, FP, TN, FN)
            perf_metrics[step] = (TPR, TNR, PPV, FPR, FNR, FDR)
        
        all_TPRs.append(TPR)
        all_FPRs.append(FPR)
        all_PPVs.append(PPV)
    
    if outfn:
        write_performance_metrics_to_file(perf_metrics, outfn)
        return prevalence, outfn, all_TPRs, all_FPRs, all_PPVs
    else:
       return prevalence, all_TPRs, all_FPRs, all_PPVs

def compute_tp_fp_tn_fn(pred_class, targets):
    """
    Compute true positive, false positive, true negative, false negative given the prediction and target
    Based on https://github.com/PyTorchLightning/metrics/blob/fa44471735f2a8216af52f18d210363f5ced9608/torchmetrics/functional/classification/stat_scores.py#L63
    
    Args:
        pred_class:  np array of bool [H,W] 
        target:      np array of binary integer [H,W] 

    Returns:
        tp:          true positive
        fp:          false positive
        tn:          true negative
        fn:          false negative
    """
    # True preidition is True when the prediction and target are the same
    # False prediction is True when the prediction and target are different
    true_pred, false_pred = targets == pred_class, targets != pred_class

    # Positive prediction is True when the prediction is 1
    # Negative prediction is True when the prediction is 0
    pos_pred, neg_pred = pred_class == 1, pred_class == 0

    # True predition * Positive prediction 
    tp = (true_pred * pos_pred).sum()
    # False predition * Positive prediction 
    fp = (false_pred * pos_pred).sum()
    # True predition * Negative prediction
    tn = (true_pred * neg_pred).sum()
    # False predition * Negative prediction
    fn = (false_pred * neg_pred).sum()

    return tp, fp, tn, fn

def calc_sensitivity(TP, FN): # TPR
    if (TP == 0.) and (FN == 0.):
        return 0.
    sensitivity = float(TP)/(float(TP) + float(FN))
    return sensitivity


def calc_specificity(TN, FP): # TNR
    if (TN == 0.) and (FP == 0.):
        return 0.
    specificity = float(TN)/(float(TN) + float(FP))
    return specificity


def calc_precision(TP, FP): # PPV
    if (TP == 0.) and (FP == 0.):
        return 0.
    ppv = float(TP)/(float(TP) + float(FP))
    return ppv


def calc_FPR(FP, TN): # FPR
    if (FP == 0) and (TN == 0):
        return 0.
    FPR = float(FP)/(float(FP) + float(TN))
    return FPR


def calc_FNR(FN, TP): # FNR
    if (FN == 0) and (TP == 0):
        return 0.
    FNR = float(FN)/(float(FN) + float(TP))
    return FNR


def calc_FDR(FP, TP): # FDR
    if (FP == 0) and (TP == 0):
        return 0.
    FDR = float(FP)/(float(FP) + float(TP))
    return FDR

def calc_performance_metrics(TP, FP, TN, FN):
    """
    Compute performance metrics from TP, FP, TN, FN

    Args:
        TP (int): Number of True Positives
        FP (int): Number of False Positives
        TN (int): Number of True Negatives
        FN (int): Number of False Negatives

    Returns:
        TPR (float): True Positive Rate
        TNR (float): True Negative Rate
        PPV (float): Positive Predictive Value
        FPR (float): False Positive Rate
        FNR (float): False Negative Rate
        FDR (float): False Discovery Rate
    
    """
    TPR = calc_sensitivity(TP, FN) # sensitivity
    TNR = calc_specificity(TN, FP) # specificity
    PPV = calc_precision(TP, FP) # precision
    FPR = calc_FPR(FP, TN) # FPR
    FNR = calc_FNR(FN, TP) # FNR
    FDR = calc_FDR(FP, TP) # FDR
    return TPR, TNR, PPV, FPR, FNR, FDR
    

def write_performance_metrics_to_file(performance, outfn):
    """
    Write performance metrics to file

    Args:
        performance (dict): dictionary of performance metrics
        outfn (str): output file name
    Returns: 
        None
    
    """
    header = ['step_size', 'TPR', 'TNR', 'PPV', 'FPR', 'FNR', 'FDR']
    od = OrderedDict(sorted(performance.items()))
    with open(outfn, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(header)
        for key,metrics in od.items():
            writer.writerow(['{0:.2f}'.format(key), '{0:.3f}'.format(metrics[0]), '{0:.3f}'.format(metrics[1]), '{0:.3f}'.format(metrics[2]), 
                             '{0:.3f}'.format(metrics[3]), '{0:.3f}'.format(metrics[4]), '{0:.3f}'.format(metrics[5])])
    return