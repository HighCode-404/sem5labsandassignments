# CAMPUS_SECTION_SRN_Lab3.py
import torch
import pandas as pd
import numpy as np


def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    target_col = tensor[:, -1]   # 
    uniq_class, class_counts = torch.unique(target_col, return_counts=True)
    prob = class_counts.float() / class_counts.sum()
    entropy = -torch.sum(prob * torch.log2(prob))
    return entropy.item()   # 
    # raise NotImplementedError  # 

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    """
    samples = tensor.shape[0]
    unq_vals = torch.unique(tensor[:, attribute])
    avg = 0.0

    for val in unq_vals:
        sub = tensor[tensor[:, attribute] == val]
        weight = sub.shape[0] / samples
        entropy_subset = get_entropy_of_dataset(sub)
        avg += weight * entropy_subset
    return avg
    # raise NotImplementedError  # 

def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    entropy = get_entropy_of_dataset(tensor)  # 
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = entropy - avg_info
    return round(info_gain.item(), 4)  # 
    # raise NotImplementedError  # 

def get_selected_attribute(tensor: torch.Tensor):
    """
    Select best attribute based on highest information gain.
    Returns (dict of info gains, best attribute index)
    """
    info_gains = {}
    for i in range(tensor.shape[1] - 1):  # exclude target column
        info_gains[i] = get_information_gain(tensor, i)
    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
    # raise NotImplementedError  # 
