from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
import pickle
import pandas as pd
from utils import DataIterator
from train_fine_tune import softmax
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

            ori_vector = graph.get_tensor_by_name('embedding/ori_vector:0')
            ours_vector = graph.get_tensor_by_name('relation/my_vector/mul:0')

            def run_predict(feed_dict):
                return session.run([ori_vector,ours_vector], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _input_relation, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    ori_vector_list = []
    ours_vector_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, label_list, seq_length in tqdm.tqdm(test_iter):

        ori_vector, ours_vector = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, label_list, 1, False))
                 )
        )
        ours_vector_list.extend(ours_vector)
        ori_vector_list.extend(ori_vector)

    with open('ours_vector.pickle','wb') as f1:
        pickle.dump(ours_vector_list,f1)
    with open('ori_vector_list.pickle','wb') as f2:
        pickle.dump(ori_vector_list,f2)

if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'processed_data/new_test_df.csv', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')


    set_test(dev_iter, config.checkpoint_path)
