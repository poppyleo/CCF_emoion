from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
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
    # save_weight=np.array([7033098.17779718,6931841.72587788,8007671.95743193]) 0.734
    # save_weight=np.array([206.14494324,218.97923942,220.27413148]) 0.736
    # save_weight=np.array([1.10,1,1.15])#0.732
    # save_weight=np.array([197806.00908617,195322.74534791,188426.1534207 ])#0.7377   8-10
    # save_weight=np.array([3.70809258, 3.25755555,3.73679014])#    6-8 #0.730
    # save_weight=np.array([151401.52489294,159212.96380022,160709.32607015])  #4-6   0.7361#   2-4w 0.736
    # save_weight=np.array([58.57988441,62.43716454,59.03211109])  #全部数据  0.7360#
    # save_weight=np.array([11026.605081,11764.85630307,11264.20511539])  #2-6+8-10w  0.7366#
    # save_weight=np.array([0.97692216,0.96566318,0.93727804])  #8-10  0.7371#
    # save_weight=np.array([1.00593857,1.01004726,0.97551633])  #8-10  0.73760
    # save_weight=np.array([1.02306123,0.97305822,0.93822369])  #8-10    0.73838
    # save_weight=np.array([1.02246141,0.97250561,0.93808773])  #8-10     0.73846483
    # save_weight=np.array([1.03936353,0.98438826,0.95084405])  #8-10     0.73862630000
    save_weight=np.array([1.03283219,0.97672083,0.94315084])  #8-10     0.73921829 best4wzl 0.728
    # save_weight = np.array([1.03403679, 0.97984169, 0.94320484])  # 8-10      0.736
    # save_weight=np.array([1.03093707,0.97680292,0.94451626])  #8-10     0.73884684000
    # save_weight = np.array([1.03203219, 0.97672083, 0.94255084])  # 8-10      0.7384
    # save_weight = np.array([1.03283219, 0.97672083, 0.94255084])  # 8-10      0.7385
    save_weight=np.array([1.,1.,1.])  #8-10     0.733
    # save_weight=np.array([1.26505346,1.11213371,1.14525734])  #8-10     0.733
    # save_weight=np.array([9.078742,8.19322796,8.23390044])  #8-10     0.733
    # save_weight=np.array([1.00053385,1.06875655,1.37497127])  #8-10     0.732
    # save_weight=np.array([1.25166652, 1.44772546,1.82606110])  #8-10     hirebert_base0.732
    # save_weight=np.array([0.98995255,0.99461485,0.99932381])  # albert权重
    # save_weight=np.array([8.73780413, 8.97501316,9.35093796])  # albert权重
    # save_weight=np.array([6.71504029,5.53606226,7.08116178])  #  ernie权重

    # save_weight=np.array([1.12242732,0.89562725,1.02147562])  #8-10      #model_0.7172_0.6931-2500
    # save_weight=np.array([1.12472216,0.8925591 ,1.01305782])  #8-10       0.7305  #model_0.7172_0.6931-2500
    # save_weight=np.array([2.38176410,2.45113634,2.5055290])  #继续训练9w步    base0.7297
    # save_weight=np.array([3.35805479, 3.14881398,3.23834057])  #继续训练6w步  0.729   base#model_0.7074_0.7147-2500
    # save_weight=np.array([4.79644774, 4.85215691,4.94857748])  #继续训练6w步  0.729   base#model_0.7074_0.7147-2500

    # save_weight=np.array([208.77808136,220.20095875,248.73359164])  #s screen_dev46 # ori   0.727 train_alldata1gru Bigru dev-2-4
    # save_weight=np.array([3.18017925,3.11458781,2.49666898])  #2-4    0.7288 model_0.7952_0.8133-3614
    # save_weight=np.array([2.58602770,2.28554999,2.22255931])  #2-4    0.73232   model_0.7952_0.8133-3614
    # save_weight=np.array([3.16616952,3.07573148,2.58647388])  #2-4    0.7297  model_0.7952_0.8133-3614
    # save_weight=np.array([3.19996691, 3.22787527,2.58401788])  #2-4     model_0.7952_0.8133-3614

    # save_weight=np.array([9.73393684,9.00150189,10.6021753])  #8-10     0.722

    # save_weight=np.array([4.86023151, 3.90833949,4.26509030])  #8-10     0.73316646000

    # save_weight=np.array([11.8184215,8.00590126,9.88657142])  #8-10     0.733
    # save_weight=np.array([1677852.01058132,1676414.79468057,1865200.39404709])     #0.7279 #8-10     0.733
    # save_weight=np.array([0.99163807,0.99469974,1.11963106])  #8-10     0.7288  两层双向Gru base_0.733
    # 0.996      1.125
    # # save_weight=np.array([2.88834897,2.87010092, 3.10570342,])  #8-10     0.7307  两层双向Gru base_0.733
    # # save_weight=np.array([691.6308901, 1.61654229,1.80558413])  #8-10      0.731     两层双向Gru base_0.733
    # save_weight=np.array([1.03383219,0.97672083,0.94355084])  #8-10      0.7385     两层双向Gru base_0.733


    # save_weight=np.array([2.26840641,2.36749361,2.77858624])  #
    # save_weight=np.array([1.27271198,1.26373634,1.41692100])  #8
    # save_weight=np.array([3.84287145,3.40278886,3.94777919])  #8
    # save_weight=np.array([5.19176044,5.13800258,5.39361413])  #8
    # save_weight=np.array([8.94934183,8.85223689, 10.6632242])  #8
    # save_weight=np.array([6.4371007,5.83821443,6.31797676])  #8 0.733继续训练
    # save_weight=np.array([6.4177278,5.82832819,6.32421361])  #8 0.7343继续训练


    # save_weight=np.array([50.10820108,50.62096942,52.12299016])  #8-10    0.731    两层双向Gru base_0.733
    # save_weight=np.array([[0.99100796,1.01840356,1.01730438]])  #8-10    0.732    两层双向Gru base_0.733
    pred_logit_list = save_weight * np.array(pred_logit_list)
    pred_label_list = np.argmax(pred_logit_list,axis=1)-1
    # print(len(pred_label_list))
    # print(len(true_label_list))
    # print(pred_label_list)
    # print(true_label_list)

    df = pd.DataFrame()
    test_result_pd = pd.read_csv(result_data_dir + 'processed_data/new_test_df.csv')
    test_result_pd['微博id'] = test_result_pd['微博id'].apply(lambda x: str(x))
    df['id'] = test_result_pd['微博id'].tolist()
    df['y'] = pred_label_list
    # df['pred_label'] = pred_label_list
    # df['text'] = test_result_pd['微博中文内容']
    # df['label'] =0
    model_name = config.checkpoint_path.split('/')[-1]
    print(model_name)
    df.to_csv(result_data_dir + model_name + '_test_result{}.csv'.format(save_weight[0]), index=False, encoding='utf-8')

    # """
    """
    融合所需参数保存
    """
    if 'test' in dev_iter.data_file:
        result_detail_f = 'test_result_detail_{}_{}_{}.json'.format(config.checkpoint_path.split('/')[-1],save_weight,config.fold)
    else:
        result_detail_f = 'dev_result_detail_{}_{}_{}.json'.format(config.checkpoint_path.split('/')[-1],save_weight,config.fold)#模型结构命名

    with open(config.ensemble_source_file + result_detail_f, 'w', encoding='utf-8') as detail:
        for idx in range(len(pred_label_list)):
            item = {}
            item['label_prob'] = pred_logit_list[idx]
            item['label']=pred_label_list[idx]
            detail.write(json.dumps(item, ensure_ascii=False, cls=NpEncoder) + '\n')

if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'processed_data/new_test_df.csv', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
