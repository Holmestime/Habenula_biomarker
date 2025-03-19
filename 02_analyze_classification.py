# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
from scipy.io import loadmat
import numpy as np
from scipy.stats import ranksums


def analysis_acc_confidence():
    """

    :return:
    """
    model_name_list = ['lstm', 'gru', 'trans']
    # check the name list order
    deep_names_list = ["LSTM (Orig.)", "LSTM (Aug.)", "LSTM (Embed.)", "GRU (Orig.)", "GRU (Aug.)", "GRU (Embed.)", "Trans (Orig.)", "Trans (Aug.)", "Trans (Embed.)"]

    # load the deep learning model performance
    deep_accuracy_list = []
    i = 0
    for c_name in model_name_list:
        for c_embedding in [None, 300]:
            for c_aug in [False, True]:
                if c_embedding is not None and c_aug:
                    continue
                # check model name
                print(f"{c_name}, embedding, {c_embedding}, aug {c_aug}")
                print(deep_names_list[i])
                i = i + 1

                data_path = f"./data/{c_name}_performance_embed_{c_embedding}_aug_{c_aug}.mat"
                data = loadmat(data_path)

                accuracy = data['acc']
                accuracy = np.squeeze(accuracy)
                deep_accuracy_list.append(accuracy)

    deep_accuracy = np.array(deep_accuracy_list)

    # Create a list to store the measure labels
    measure_labels = ['Accuracy','Specificity','Sensitivity','F1-Score','AUC']
    tradition_name_list = ['LDA','LR','SVM','MLP','RF','AdaBoost']

    data_file = "./data/ml_performance.npy"
    # 4 * 4 * 1000
    data_list = np.load(data_file)
    # 5 measure * 6 classifier * 1000 times
    data_list = np.array(data_list)
    # get acc
    traditional_accuracy = data_list[0, :, :]


    total_acc = np.concatenate([traditional_accuracy, deep_accuracy], axis=0)
    total_name = tradition_name_list + deep_names_list

    acc_mean = np.mean(total_acc,axis=1)
    sort_indice = np.argsort(acc_mean)[::-1]

    sort_acc_mean = np.sort(acc_mean)[::-1]
    sort_total_acc = total_acc[sort_indice, :]
    sort_name = [total_name[i] for i in sort_indice]

    # 计算每个类别的均值和标准差
    means = np.mean(sort_total_acc, axis=1)
    std_devs = np.std(sort_total_acc, axis=1)

    # 创建 DataFrame 用于绘图
    df = pd.DataFrame({
        'Name': sort_name,
        'Mean Accuracy': means,
        'Std Dev': std_devs
    })

    plt.figure(figsize=(7 / 2.54, 5.85 / 2.54,))
    # 绘制带 errorbar 的 barplot
    # plt.figure(figsize=(10, 6))
    sns.barplot(x='Name', y='Mean Accuracy', data=df, yerr=df['Std Dev'], palette='tab20', capsize=0.2)
    plt.xlabel("")
    plt.ylabel('Average accuracy')
    plt.xticks(rotation=45, ha='right')  # ha='right' 确保旋转后的标签右对齐
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.savefig('../plot_pdf/pdf/acc_new.pdf', dpi=300, format='pdf')
    plt.savefig('../plot_pdf/png/acc_new.png', dpi=300, format='png')
    plt.show()
    return



def significance_test():
    """

    :return:
    """
    model_name_list = ['lstm', 'gru', 'trans']
    # check the name list order
    deep_names_list = ["LSTM (Orig.)", "LSTM (Aug.)", "LSTM (Embed.)", "GRU (Orig.)", "GRU (Aug.)", "GRU (Embed.)", "Trans (Orig.)", "Trans (Aug.)", "Trans (Embed.)"]

    # load the deep learning model performance
    deep_accuracy_list = []
    i = 0
    for c_name in model_name_list:
        for c_embedding in [None, 300]:
            for c_aug in [False, True]:
                if c_embedding is not None and c_aug:
                    continue
                print(f"{c_name}, embedding, {c_embedding}, aug {c_aug}")
                print(deep_names_list[i])
                i = i + 1


                data_path = f"./data/{c_name}_performance_embed_{c_embedding}_aug_{c_aug}.mat"
                data = loadmat(data_path)

                c_performance = [data['acc'], data['spec'], data['sen'], data['f1'], data['auc']]

                c_performance = np.concatenate(c_performance)

                deep_accuracy_list.append(c_performance)

    deep_accuracy = np.array(deep_accuracy_list)



    # Create a list to store the measure labels
    measure_labels = ['Accuracy','Specificity','Sensitivity','F1-Score','AUC']
    tradition_name_list = ['LDA','LR','SVM','MLP','RF','AdaBoost']

    data_file = "./data/ml_performance.npy"
    # 4 * 4 * 1000
    data_list = np.load(data_file)
    # 5 measure * 6 classifier * 1000 times
    data_list = np.array(data_list)
    # get acc

    data_list = np.moveaxis(data_list, 1, 0)

    traditional_accuracy = data_list
    total_acc = np.concatenate([traditional_accuracy, deep_accuracy], axis=0)


    total_name = tradition_name_list + deep_names_list

    acc_mean = np.mean(total_acc[:,0,:],axis=1)
    sort_indice = np.argsort(acc_mean)[::-1]

    sort_acc_mean = np.sort(acc_mean)[::-1]
    # 15 classifiers, 5 metric, 1000 times
    sort_total_acc = total_acc[sort_indice, :]
    sort_name = [total_name[i] for i in sort_indice]

    #

    data = sort_total_acc

    print(f'---------------------------------------------------------------------------significance test')
    # Select the data for the first classifier and the other classifiers
    first_classifier = data[1,:, :]
    other_index = [0] + [i for i in range(2, 15)]
    other_classifiers = data[np.array(other_index), :, :]
    other_name = [sort_name[i] for i in other_index]

    # Perform the Wilcoxon rank sum test for each measurement
    p_values = []
    compare_times = other_classifiers.shape[0] * other_classifiers.shape[1]
    alpha = 0.05
    # Apply Bonferroni correction
    corrected_alpha = alpha / compare_times

    # metric dimension
    for i in range(first_classifier.shape[0]):
        print(f'--------------Estimate performance-{measure_labels[i]}')
        first_classifier_measurement = first_classifier[i]
        other_classifiers_measurement = other_classifiers[:, i, :]

        # Perform the Wilcoxon rank sum test for each other classifier
        for j in range(other_classifiers.shape[0]):
            print(f'LR compare to {other_name[j]}')
            other_classifier_measurement = other_classifiers_measurement[j, :]
            statistic, p_value = ranksums(first_classifier_measurement, other_classifier_measurement)
            p_values.append(p_value)

            if p_value < corrected_alpha:
                # print(f"Significant (p-value = {p_value})")
                print("")
            else:
                print(f"Not significant (p-value = {p_value})")


if __name__ == '__main__':
    config = {

        "font.family": 'Arial',
        'font.size': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    rcParams.update(config)

    # analysis_acc_confidence()
    significance_test()