class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 3
        self.over_sample = True

        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 20 * 3

        self.train_epoch =2
        self.sequence_length = 150

        self.learning_rate = 1e-4
        self.embed_learning_rate = 2e-5
        self.batch_size =64
        self.embed_trainable = True

        self.as_encoder = True


        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file  = '/data/wangzhili/lei/ensemble/sourcefile_best/'
        self.ensemble_result_file = '/data/wangzhili/lei/ensemble/resultfile/'

        self.checkpoint_path = "/home/none404/hm/lei/data/emotion/Savemodel/runs_2/1590136016/model_0.7165_0.6902-1250"


        self.pretrain_model = 'roberta_base'
        self.school=True
        # self.school=False

        if self.school:
            """学校"""
            self.data_process = '/home/wangzhili/data/emotion/'#学校
            self.save_model = '/home/wangzhili/data/emotion/Savemodel/'#学校

            if self.pretrain_model=='roberta_base':
                bert_file = '/home/wangzhili/pretrained_model/roberta_zh_l12/'  # base
                # bert_file ='/home/wangzhili/pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/' #roberta_wwm
                self.bert_file = bert_file+'bert_model.ckpt'
                # self.bert_file = '/home/wangzhili/data/ccf_emotion/roberta_model/model.ckpt-1000000' #继续训练
                self.bert_config_file = bert_file+ 'bert_config.json'
                self.vocab_file = bert_file+'vocab.txt'
                """large"""
            elif self.pretrain_model=='roberta_large':
                bert_file = '/home/wangzhili/pretrained_model/roeberta_zh_L-24_H-1024_A-16/'  # large
                self.bert_file = bert_file + 'roberta_zh_large_model.ckpt'
                # self.bert_file = '/home/wangzhili/data/ccf_emotion/roberta_model/model.ckpt-1000000' #继续训练
                self.bert_config_file = bert_file + 'bert_config_large.json'
                self.vocab_file = bert_file + 'vocab.txt'
            else:
                raise EOFError
        else:
            self.data_process = '/data/wangzhili/lei/nCoV/'  # 国双
            self.save_model = '/data/wangzhili/lei/nCoV/Savemodel/'  # 国双
            # 国双
            if self.pretrain_model == 'roberta_base':
                # self.bert_file = '/data/pengcheng01/pretrained_model/roberta_zh_l12/bert_model.ckpt'
                self.bert_file = '/data/wangzhili/lei/nCoV/Continue_model/model.ckpt-1000000'  # 继续训练
                self.bert_config_file = '/data/pengcheng01/pretrained_model/roberta_zh_l12/bert_config.json'
                self.vocab_file = '/data/pengcheng01/pretrained_model/roberta_zh_l12/vocab.txt'
            elif self.pretrain_model == 'bert_wwm':
                """BERT_WWM"""
                bert_file ='/data/pengcheng01/pretrained_model/BERT_wwm/'
                self.bert_file = bert_file + 'bert_model.ckpt'
                self.bert_config_file = bert_file + 'bert_config.json'
                self.vocab_file = bert_file + 'vocab.txt'
            elif self.pretrain_model == 'albert':
                """albert"""
                bert_file ='/data/wangzhili/lei/albert_base_zh/'
                self.bert_file = bert_file + 'albert_model.ckpt'
                self.bert_config_file = bert_file + 'albert_config_base.json'
                self.vocab_file = bert_file + 'vocab.txt'
            elif self.pretrain_model =='nezha':
                bert_file = '/data/wangzhili/lei/Model/Nezha/'
                self.bert_file = bert_file + 'model.ckpt-691689'
                self.bert_config_file = bert_file + 'bert_config.json'
                self.vocab_file = bert_file + 'vocab.txt'
            else:
                raise EOFError
        #
        # 追一4albert_large
        self.data_process = '/home/none404/hm/lei/data/emotion/'
        self.save_model = '/home/none404/hm/lei/data/emotion/Savemodel/'

        bert_file = '/home/none404/hm/Model/robert_zh_l12/'
        self.bert_file = bert_file + 'bert_model.ckpt'
        self.bert_config_file = bert_file + 'bert_config.json'
        self.vocab_file  = bert_file + 'vocab.txt'



        self.data_processed = self.data_process+'processed_data/'
        self.use_origin_bert ='dym'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'hire':Hirebert
        self.is_avg_pool =True # True: 使用平均avg_pool

        self.pool='mean'  #'mean' 为平均池化，其他为最大池化
        self.fold=9#fold=5,验证为80000-100000
        self.compare_result=False
        self.kfoldpath='Kfold_data/'
        self.result_file='result/'
        self.gpu=2
        self.keep_prob = 0.9
        #卷积参数
        self.addcnn = False   #是否+cnn
        self.num_filters=512 #卷积核个数
        self.kernel_size = 7  # 卷积核尺寸
        self.addgru=True
        self.addadv = False
        self.addfgm=True
        #hire_bert参数
        self.gru_hidden_dim=64
        self.cls=False
        #国双
        #gpu5 albert_base
        #gpu6 bert_wwm

        #学校服务器
        # gpu3 roberta large 双向gru

        self.gru=True
        self.lstm=False
        self.loss='loss'