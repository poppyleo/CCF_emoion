import os
import json
from collections import OrderedDict,Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/home/qinbin/Leipengbin/CCF_emotion")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config

config = Config()

"""
剔除的文件
"""
remove_list = []


def vote_ensemble(path, dataset, output_path, remove_list):
    """
    投票平均
    """
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ')
    for file_name in single_model_list:
        print(file_name)

    label_list = OrderedDict()
    for text_index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            print(text_index)
            print('Ensembling.....')
            print('Text File: ', file)
            with open(path + file) as f:
                for i, line in tqdm(enumerate(f.readlines())):
                    item = json.loads(line)
                    if i not in label_list:
                        label_list[i] = []
                    label_list[i].append(item['label'])

    print(len(label_list))

    print('Getting Result.....')
    all_label = np.array([])
    for key in tqdm(label_list.keys()):
        label = np.array(label_list[key])
        all_label = np.concatenate((all_label, label))
    vote_list = []
    all_label = all_label.reshape(len(label_list), -1)
    # print(all_label.T)
    all_label = all_label.astype(int)
    for _label in all_label:
        vote_label = Counter(_label.tolist()).most_common()[0][0]
        vote_list.append(vote_label)
    df = pd.DataFrame()
    test_result_pd = pd.read_csv(config.data_processed  + 'test.csv')
    df['id'] = test_result_pd['微博id'].tolist()
    df['y'] = vote_list
    df.to_csv(output_path + 'test_resultvote.csv', encoding='utf-8', index=False)


def score_average_ensemble(path, dataset, output_path, remove_list):
    """
    概率平均
    """
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ')
    for file_name in single_model_list:
        print(file_name)

    logits_list = OrderedDict()
    for text_index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            print(text_index)
            print('Ensembling.....')
            print('Text File: ', file)
            with open(path + file) as f:

                for i, line in tqdm(enumerate(f.readlines())):
                    item = json.loads(line)
                    if i not in logits_list:
                        logits_list[i] = []
                    np.array(item['label_prob'])
                    logits_list[i].append(item['label_prob'])

    print(len(logits_list))

    print('Getting Result.....')
    all_logits = []
    for key in tqdm(logits_list.keys()):
        logits = np.zeros_like(np.array(logits_list[key][0]))
        for i in logits_list[key]:
            logits += np.array(i)
        logits_label = np.argmax(logits) - 1
        all_logits.append(logits_label)
    # print(all_logits)
    #
    df = pd.DataFrame()
    test_result_pd = pd.read_csv(config.data_processed  + 'test.csv')
    df['id'] = test_result_pd['微博id'].tolist()
    df['y'] = all_logits
    df.to_csv(output_path + 'test_resultscore.csv', encoding='utf-8',index=False)


if __name__ == '__main__':
    remove_list = [

    ]
    # 测试集
    score_average_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)
    vote_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)

    # 验证集
    # vote_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)
    # score_average_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)

