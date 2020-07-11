# import tensorflow as tf
# import numpy as np
# from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # BERT

# # from tf_utils.eltra_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # Electra
# # from tf_utils.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # ALBERT
# from tensorflow.contrib.layers.python.layers import initializers
# from tf_utils.crf_utils import rnncell as rnn
from config import Config
config = Config()
import tensorflow as tf
import numpy as np
# from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # BERT
if config.pretrain_model == 'albert':
    from tf_utils.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # ALBERT
elif config.pretrain_model =='nezha':
    from tf_utils.nezha import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # Nezha
    # from tf_utils.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # ALBERT
else:
    from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # BERT
from tensorflow.contrib.crf import crf_log_likelihood,  viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tf_utils.crf_utils import rnncell as rnn
from tf_utils.bert_modeling import layer_norm
from  focal_loss import focal_loss

# import memory_saving_gradients
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# # 对于CRF这种多优化目标的层，memory_saving_gradients会出bug，注释即可。



class Model:

    def __init__(self, config):
        self.config = config
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.label = tf.placeholder(tf.int32, [None], name='label')  # 情感标签
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')  # 分段标签
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')

        self.relation_num = config.relation_num
        self.initializer = initializers.xavier_initializer()

        self.init_embedding(bert_init=True)
        output_layer = self.word_embedding
        gru_num=int(self.config.embed_dense_dim/2)
        hidden_size = get_shape_list(output_layer)[-1]
        if config.addcnn:
            ##bert后接text CNN
            num_filters = config.num_filters
            kernel_size = config.kernel_size  # 卷积核尺寸
            output_layer = tf.layers.conv1d(output_layer, num_filters, kernel_size) #[batch,(seq_length - kernel_size + 1）,num_filters]
            hidden_size = num_filters
            pool_size = config.sequence_length - kernel_size + 1
            # 每个卷积核得到一个（seq_length - kernel_size + 1）size的向量
        elif config.addgru:
            GRU_cell_fw = tf.contrib.rnn.GRUCell(gru_num,name='fw_tr')  # 参数可调试
        # 后向
            GRU_cell_bw = tf.contrib.rnn.GRUCell(gru_num,name='bw_tr')  # 参数可调试
            output_layer_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=output_layer,
                                                                                 sequence_length=None,
                                                                                 dtype=tf.float32)[0]
            output_layer_1=tf.concat([output_layer_1[0],output_layer_1[1]],axis=-1)
            # GRU_cell_fw_1 = tf.contrib.rnn.GRUCell(gru_num, name='fw_tr',reuse=True)  # 参数可调试
            # # 后向
            # GRU_cell_bw_1 = tf.contrib.rnn.GRUCell(gru_num, name='fw_tr',reuse=True)  # 参数可调试
            # output_layer_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw_1,
            #                                                cell_bw=GRU_cell_bw_1,
            #                                                inputs=output_layer_1,
            #                                                sequence_length=None,
            #                                                dtype=tf.float32)[0]
            # output_layer_2 = tf.concat([output_layer_2[0], output_layer_2[1]], axis=-1)
            # # output_layer_3 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
            # #                                                  cell_bw=GRU_cell_bw,
            # #                                                  inputs=output_layer_2,
            # #                                                  sequence_length=None,
            # #                                                  dtype=tf.float32)[0]
            # # output_layer_3 = tf.concat([output_layer_3[0], output_layer_3[1]], axis=-1)
            # output_layer  = tf.concat([output_layer_1,output_layer_2],axis=-1)
            output_layer =output_layer_1
            pool_size = config.sequence_length
            hidden_size = output_layer.shape[-1]

        else:
            pool_size=config.sequence_length

        # 池化+drop_out
        if self.config.is_avg_pool:
            if config.pool=='mean':
                avpooled_out = tf.layers.average_pooling1d(output_layer, pool_size=pool_size, strides=1)  # shape = [batch, num_filters]
            elif config.pool=='join':
                avpooled_out = tf.layers.average_pooling1d(output_layer, pool_size=pool_size, strides=1)  # shape = [batch, num_filters]
                maxpooled_out = tf.layers.max_pooling1d(output_layer, pool_size=config.sequence_length, strides=1)
                avpooled_out = tf.concat([avpooled_out,maxpooled_out],axis=-1)
                hidden_size = 2*hidden_size
            else:
                avpooled_out = tf.layers.max_pooling1d(output_layer,pool_size=config.sequence_length,strides=1)
            avpooled_out = tf.reshape(avpooled_out, [-1, hidden_size])
        else:
            avpooled_out = output_layer[:, 0:1, :]  # pooled_output
            avpooled_out = tf.squeeze(avpooled_out, axis=1)


        def logits_and_predict(num_classes, name_scope=None):
            with tf.name_scope(name_scope):
                inputs = tf.nn.dropout(avpooled_out, keep_prob=config.keep_prob,name = 'my_vector')  #分类前一个维度向量
                print('*'*88)
                print(inputs)
                # inputs=avpooled_out
                logits = tf.layers.dense(inputs, num_classes,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name='logits')
                predict = tf.round(tf.sigmoid(logits), name="predict")
            return logits, predict

        self.logits, self.predict = logits_and_predict(self.relation_num, name_scope='relation')
        # print(self.logits)
        self.one_hot_labels = tf.one_hot(self.label, depth=self.relation_num, dtype=tf.float32, name="one_hot_label")
        if config.loss=='focal_loss':
            # self.loss =focal_loss(self.one_hot_labels,self.logits,[16902,25392,57619])
            self.loss =focal_loss(self.logits,self.one_hot_labels)
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_hot_labels, logits=self.logits, name='losses')
            self.loss = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name='loss')


    def init_embedding(self, bert_init=True):
        with tf.name_scope('embedding'):
            word_embedding = self.bert_embed(bert_init)
            print('self.config.embed_dense_dim:', self.config.embed_dense_dim)
            word_embedding = tf.layers.dense(word_embedding, self.config.embed_dense_dim, activation=tf.nn.relu)
            hidden_size = word_embedding.shape[-1].value
        self.word_embedding = word_embedding
        print(word_embedding.shape)
        self.output_layer_hidden_size = hidden_size

    def bert_embed(self, bert_init=True):
        bert_config_file = self.config.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word,
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)
        offical_vector = model.get_pooled_output()
        offical_vector = tf.reshape(offical_vector,[-1,offical_vector.shape[-1]],name='ori_vector')
        layer_logits = []
        for i, layer in enumerate(model.all_encoder_layers):
            layer_logits.append(
                tf.layers.dense(
                    layer, 1,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    name="layer_logit%d" % i
                )
            )#[batch_size]

        layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        layer_dist = tf.nn.softmax(layer_logits)
        seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        pooled_output = tf.squeeze(pooled_output, axis=2)
        pooled_layer = pooled_output

        char_bert_outputs = pooled_layer

        A_output =pooled_layer
        R_output = model.get_sequence_output()
        RoA = tf.multiply(R_output, A_output)
        R_A = tf.add(R_output, A_output)
        fill_output = tf.concat([R_output, A_output, RoA, R_A],
                                axis=2)  # 【batch_size,seq_len,hidden_size*4】后面会借一个enbed_dense_dim的全连接

        # """Hire_bert"""

        def BidirectionalGRUEncoder(hidden_dim,inputs,name):
            # 双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，然后在经过Attention层，将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
            # 输入inputs的shape是[batch_size, max_len, hidden_size]
            #参数共享
            with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
                #
                #前向
                GRU_cell_fw = tf.contrib.rnn.GRUCell(hidden_dim) #参数可调试
                #后向
                GRU_cell_bw = tf.contrib.rnn.GRUCell(hidden_dim) #参数可调试
                ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                     cell_bw=GRU_cell_bw,
                                                                                     inputs=inputs,
                                                                                     sequence_length=None,
                                                                                     dtype=tf.float32)
                outputs = tf.concat((fw_outputs, bw_outputs), 2)
                return outputs
        layer_logits = []
        # 得到每一层的alpha
        #共享参数

        for i, layer in enumerate(model.all_encoder_layers):
            ###Bigru是为了确定每一层的权重
            #两层双向GRU
            B_1 = BidirectionalGRUEncoder(self.config.gru_hidden_dim, layer,'bigru_1')  # 结果是前向gru和后向gru
            B_2= BidirectionalGRUEncoder(self.config.gru_hidden_dim, B_1,'bigru2')
            #将四个方向的向量拼接到一起
            U_layer=tf.concat((B_1,B_2), 2)
            layer_logits.append(
                tf.layers.dense(
                    U_layer, 1,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    name="addgrulayer_logit%d" % i
                )
            )
        # 得到每一层的alpha
        layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        layer_dist = tf.nn.softmax(layer_logits) #权重
        seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        A_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        A_output = tf.squeeze(A_output, axis=2)#
        R_output=model.get_sequence_output()
        RoA=tf.multiply(R_output, A_output)
        R_A=tf.add(R_output,A_output)
        Mire_output=tf.concat([R_output,A_output,RoA,R_A],axis=2)   #【batch_size,seq_len,hidden_size*4】后面会借一个enbed_dense_dim的全连接

        if self.config.use_origin_bert=='ori':
            final_hidden_states = model.get_sequence_output()  # 原生bert
            self.config.embed_dense_dim = 768
        elif self.config.use_origin_bert=='hire':
            final_hidden_states = Mire_output  # hirebert
        elif self.config.use_origin_bert=='dym':
            final_hidden_states = char_bert_outputs  # 多层融合bert
            self.config.embed_dense_dim = 512
        elif self.config.use_origin_bert == 'fill_bert':
             final_hidden_states = fill_output  # 多层融合bert+互补
        else:
            raise SyntaxError# print('输入的参数错误') config.use_origin_bert


        tvars = tf.trainable_variables()
        init_checkpoint = self.config.bert_file  # './chinese_L-12_H-768_A-12/bert_model.ckpt'
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if bert_init:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))
        print('init bert from checkpoint: {}'.format(init_checkpoint))
        return final_hidden_states
