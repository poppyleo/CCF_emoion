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

result_data_dir = Config().data_process
gpu_id = Config().gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)


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
    # with open(model_file+'_weight.txt','r') as f:
    #     content=f.readline().strip('\n')
    #     content=content.replace('[','')
    #     content=content.replace(']','')
    #     save_weight=content.split(' ')
    # # save_weight=np.array(save_weight).astype(float)
    # save_weight=np.array([7033098.17779718,6931841.72587788,8007671.95743193])
    # save_weight=np.array([206.14494324,218.97923942,220.27413148])
    print('改变权重前的F1为：',f1_score(true_label_list, pred_label_list, average='macro'))
    op = OptimizedF1()
    op.fit(np.array(pred_logit_list), true_label_list)
    print('改变权重后的F1为：',op.predict(np.array(pred_logit_list), true_label_list))
    pred_logit_list = op.coefficients() * np.array(pred_logit_list)
    print(op.coefficients())
    # print(len(pred_label_list))
    # print(len(true_label_list))
    # print(pred_label_list)
    # print(true_label_list)
    # save_weight = np.array([1.03283219, 0.97672083, 0.94315084])
    # pred_logit_list = save_weight * np.array(pred_logit_list)
    # pred_label_list = np.argmax(pred_logit_list, axis=1) - 1
    # dev_df=pd.read_csv(config.data_process + 'processed_data/new_dev_df.csv',encoding='utf_8_sig')
    # dev_df['pred_label']=pred_label_list
    # dev_df.to_csv(config.data_process+'compare_result.csv',index=False,encoding='utf_8_sig')



if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=config.data_process + 'processed_data/new_dev_df.csv',use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
