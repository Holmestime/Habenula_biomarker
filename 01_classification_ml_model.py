# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_all_feature_data
from sklearn.linear_model import LogisticRegression
from matplotlib import rcParams
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import seaborn as sns
import random
from sklearn.pipeline import Pipeline
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from scipy.stats import sem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import ranksums
from scipy.io import loadmat


def load_all_feature_data():
    """
    Get the data of all features: trial_num * 24

    :return:
    """
    feature_file = "./data/ml_data.mat"
    file_handle = loadmat(feature_file)
    features = file_handle["features"]
    label = file_handle["labels"]
    # 1 depression, 0 normal
    return features, label

def get_performance(data, label, n_repeats=200):
    """

    :param data:
    :param label:
    :return:
    """
    scoring = {
        'Accuracy': make_scorer(accuracy_score),
        'Specificity': make_scorer(recall_score, pos_label=0),
        'Sensitivity': make_scorer(recall_score, pos_label=1),
        'F1-Score': make_scorer(f1_score),
        'AUC': make_scorer(roc_auc_score)
    }

    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'LR': LogisticRegression(class_weight='balanced'),
        'SVM': SVC(),
        'MLP': MLPClassifier(hidden_layer_sizes=(24,24),max_iter=500, learning_rate='adaptive'),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)

    c_name = []
    c_acc = []
    c_spec = []
    c_sen = []
    c_f1 = []
    c_auc = []

    for clf_name, clf in classifiers.items():
        pipeline = make_pipeline(StandardScaler(), clf)
        scores = cross_validate(pipeline, data, label, cv=cv, scoring=scoring)

        print("Classifier:", clf_name)
        print("Accuracy:", scores['test_Accuracy'].mean())
        print("Specificity:", scores['test_Specificity'].mean())
        print("Sensitivity:", scores['test_Sensitivity'].mean())
        print("F1-Score:", scores['test_F1-Score'].mean())
        print("AUC:", scores['test_AUC'].mean())
        print("------------------------")

        c_name.append(clf_name)
        c_acc.append(scores['test_Accuracy'])
        c_spec.append(scores['test_Specificity'])
        c_sen.append(scores['test_Sensitivity'])
        c_f1.append(scores['test_F1-Score'])
        c_auc.append(scores['test_AUC'])

    acc = np.array(c_acc)
    spec = np.array(c_spec)
    sen = np.array(c_sen)
    f1 = np.array(c_f1)
    auc = np.array(c_auc)

    result = [acc, spec, sen, f1, auc]
    np.save("./data/ml_performance.npy", result)

if __name__ == '__main__':
    data, label = load_all_feature_data()
    label = label.ravel()
    get_performance(data, label,n_repeats=200)
