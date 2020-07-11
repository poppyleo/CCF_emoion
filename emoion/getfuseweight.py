from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
import pandas as pd
from utils import DataIterator
from train_fine_tune import softmax,OptimizedF1
from sklearn.metrics import f1_score
from functools import partial
import scipy as sp

result_data_dir = Config().data_process
gpu_id = Config().gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)

class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        coef_rate=[i/sum(coef) for i in coef]
        # print(X[1])
        # print(weight[1])
        addw_pred=[np.array(X[i])*coef_rate[i] for i in range(len(X))]
        true_x=np.argmax(sum(addw_pred),axis=1)
        ll = f1_score(y, true_x, average='macro')
        #         print(ll)
        # #         raise EOFError
        #         print(coef)
        return 1 / ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        # 猜测初始权值
        initial_coef =[1.]*len(X)

        # 最小化一个或多个变量的标量函数使新的F1最小的权值
        self.coef_ = sp.optimize.basinhopping(loss_partial, initial_coef, niter=10000,
                                              # callback=print_fun,
                                              stepsize=0.0000001)

    def predict(self, X, y):
        coef_rate=[i/sum(self.coef_['x']) for i in self.coef_['x']]
        addw_pred=[np.array(X[i])*coef_rate[i] for i in range(len(X))]
        true_x=np.argmax(sum(addw_pred),axis=1)
        return f1_score(y, true_x, average='macro')

    def coefficients(self):
        return self.coef_['x']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _input_relation = graph.get_operation_by_name("label").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]

            predict = graph.get_operation_by_name('relation/predict').outputs[0]
            logits = graph.get_operation_by_name('relation/Sigmoid').outputs[0]

            def run_predict(feed_dict):
                return session.run([logits, predict], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _input_relation, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    true_label_list = []
    pred_label_list = []
    pred_logit_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, label_list, seq_length in tqdm.tqdm(test_iter):

        logits, pred_label = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, label_list, 1, False))
                 )
        )
        pred_label =softmax(logits)
        pred_label = np.argmax(pred_label, axis=1)
        true_label_list.extend(label_list)
        pred_label_list.extend(pred_label)
        pred_logit_list.extend(softmax(logits))
    return pred_logit_list,true_label_list
    # with open(model_file+'_weight.txt','r') as f:
    #     content=f.readline().strip('\n')
    #     content=content.replace('[','')
    #     content=content.replace(']','')
    #     save_weight=content.split(' ')
    # # save_weight=np.array(save_weight).astype(float)
    # save_weight=np.array([7033098.17779718,6931841.72587788,8007671.95743193])
    # save_weight=np.array([206.14494324,218.97923942,220.27413148])
    # print('改变权重前的F1为：',f1_score(true_label_list, pred_label_list, average='macro'))
    # op = OptimizedF1()
    # op.fit(np.array(pred_logit_list), true_label_list)
    # print('改变权重后的F1为：',op.predict(np.array(pred_logit_list), true_label_list))
    # pred_logit_list = op.coefficients() * np.array(pred_logit_list)
    # print(op.coefficients())
    # print(len(pred_label_list))
    # print(len(true_label_list))
    # print(pred_label_list)
    # print(true_label_list)



if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    dev=False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size,
                            # data_file=config.data_process + 'processed_data/dev4test_df.csv',
                            data_file=result_data_dir + 'processed_data/new_test_df.csv',
                            use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    checkpoint_path_list=[
        '/data/wangzhili/nCoV/best_model/model_0.6871_0.7139-5000',#wzl 0.728  0.739
        '/data/wangzhili/lei/nCoV/Savemodel/runs_4/1585917501/model_0.7060_0.6902-2500', # 0.7307 一个gru
        '/data/wangzhili/lei/nCoV/best_model/1584697969/model_0.7145_0.7035-2500',#0.733 #两个gru
        '/data/wangzhili/lei/nCoV/best_model/1584710443/model_0.7099_0.7034-2500', #0.73448 #三个
        '/data/wangzhili/nCoV/best_model/model_0.6925_0.6945-2000'  #  0.73 [4.86023151e+47 3.90833949e+47 4.26509030e+47]
    ]
    weight_list=[
        [1.03283219, 0.97672083, 0.94315084],
        [],
        [],
        [],
        [4.86023151, 3.90833949,4.26509030]
    ]
    allpred=[]
    for idx,checkpoint_path in enumerate(checkpoint_path_list):
        pred_prob,true_label=set_test(dev_iter, checkpoint_path)
        weight=weight_list[idx]
        if weight!=[]:
            pred_prob=weight*np.array(pred_prob)
        allpred.append(pred_prob)
    if dev:
        op = OptimizedF1()
        prepred_label=np.argmax(sum(allpred),axis=1)
        print('改变权重前的F1为：', f1_score(prepred_label, true_label, average='macro'))
        print(allpred.__len__())
        op.fit(allpred, true_label)
        print(op.coefficients())
        print('改变权重后的F1为：', op.predict(allpred, true_label))
    else:
        dym_weight=[4.06029342, 1.93046457,3.24383058,1.35236854,- 0.87541554]
        coef_rate=[i/sum(dym_weight) for i in dym_weight]

        addw_pred=[np.array(allpred[i])*coef_rate[i] for i in range(len(allpred))]
        pred_x=np.argmax(sum(addw_pred),axis=1)-1

        addw_ori=[np.array(allpred[i])*1 for i in range(len(allpred))]
        pred_ori=np.argmax(sum(addw_ori),axis=1)-1

        df = pd.DataFrame()
        df_1=pd.DataFrame()
        test_result_pd = pd.read_csv(result_data_dir + 'processed_data/new_test_df.csv')
        test_result_pd['微博id'] = test_result_pd['微博id'].apply(lambda x: str(x))
        df['id'] = test_result_pd['微博id'].tolist()
        df['y'] = pred_x

        df_1['id'] = test_result_pd['微博id'].tolist()
        df_1['y'] = pred_ori

        model_name = config.checkpoint_path.split('/')[-1]
        print(model_name)
        df.to_csv(result_data_dir + model_name + '_test_result{}.csv'.format(dym_weight[0]), index=False,
                  encoding='utf-8')
        df_1.to_csv(result_data_dir + model_name + '_test_result_ori{}.csv'.format(dym_weight[0]), index=False,
                  encoding='utf-8')