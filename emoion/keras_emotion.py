import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense,LSTM, GRU,GlobalAvgPool1D,AveragePooling1D,Bidirectional,Dropout
from sklearn.metrics import f1_score
from config import Config
from adversal import adversarial_training,loss_with_gradient_penalty
import os
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

config=Config()
gpu_id = Config().gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config().gpu)

num_classes = 3
maxlen = 150
batch_size = 32
config_path = config.bert_config_file
checkpoint_path = config.bert_file
dict_path = config.vocab_file


def load_data(filename):
    df=pd.read_csv(filename,encoding='utf_8_sig')
    df['text'].fillna('',inplace=True)
    D=list(zip(list(df['text']),list(df['label'])))
    return D


# 加载数据集
data_dir = config.data_process
train_data = load_data(data_dir + 'processed_data/new_train_df.csv')
print(train_data[:2])
valid_data = load_data(data_dir + 'processed_data/new_dev_df.csv')
test_data = load_data(data_dir + 'processed_data/new_test_df.csv')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if label==-1:
                label=[1.,0.,0.]
            elif label==0:
                label=[0.,1.,0.]
            else:
                label=[0.,0.,1.]
            batch_labels.append(label)
            # batch_labels = to_categorical(batch_labels,num_classes)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model='albert',
    return_keras_model=False,
)
if config.cls:
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output) #CLS
else:
    #sequence_output
    output = Lambda(lambda x: x[:, :], name='sequence-token')(bert.model.output)  #
    if config.gru:
        output=Bidirectional(GRU(256))(output)
        print(output)
    elif config.lstm:
        output = Bidirectional(LSTM(256))(output)
    else:
        pass
    # output =GlobalAvgPool1D()(output) #平均池化
output=Dropout(0.15)(output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

if config.addadv:
    """添加扰动"""
    loss=loss_with_gradient_penalty()
else:
    loss='categorical_crossentropy'

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
    #     1000: 1,
    #     2000: 0.1
    # }),
    # metrics=['accuracy'],
    metrics=['categorical_accuracy'],
)

# 写好函数后，启用对抗训练只需要一行代码
if config.addfgm:
    adversarial_training(model, 'Embedding-Token', 0.5)

# 转换数据集
# train_generator = data_generator(train_data, batch_size)
# valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    true,pred=[],[]
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        # print(model.predict(x_true))
        y_true=y_true.argmax(axis=1)
        true.extend(y_true)
        pred.extend(y_pred)
    print(true)
    print(pred)
    F1=f1_score(true,pred)
    return F1


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_{}.weights'.format(config.gru))
        # test_acc = evaluate(test_generator)
        print(
            u'F1: %.5f, best_F1: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

df_train = pd.read_csv(config.data_process+'ks/nCoV_100k_train.labled.csv',encoding='utf_8_sig')
evaluator = Evaluator()
gkf = StratifiedKFold(n_splits=5).split(X=df_train['text'],
                                        y=df_train['label'])
test_preds = []

for fold, (train_idx, valid_idx) in enumerate(gkf):
    print('**********************fold{}**************************'.format(fold))
    train_data=df_train.iloc[train_idx]
    valid_data=df_train.iloc[valid_idx]
    train=list(zip(list(train_data['text']),list(train_data['label'])))
    valid=list(zip(list(valid_data['text']),list(valid_data['label'])))
    train_generator = data_generator(train, batch_size)
    valid_generator = data_generator(valid, batch_size)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        #     1000: 1,
        #     2000: 0.1
        # }),
        # metrics=['accuracy'],
        metrics=['categorical_accuracy'],
    )

    # 写好函数后，启用对抗训练只需要一行代码
    if config.addfgm:
        adversarial_training(model, 'Embedding-Token', 0.5)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=2,
        callbacks=[evaluator]
    )
    model.load_weights('best_model_{}.weights'.format(config.gru))
    i=0
    for x_true,y_true in test_generator:
        y_pred = model.predict(x_true)
        if i==0:
            t_pred=y_pred
            i = 1
        else:
            t_pred=np.concatenate([t_pred,y_pred])
    test_preds.append(t_pred)

sub = np.average(test_preds, axis=0)
sub = np.argmax(sub,axis=1)-1

# df_sub['y'] = sub-1
# df_sub['id'] = df_sub['id']
# df_sub.to_csv('test_sub.csv',index=False, encoding='utf-8')
# #     pred = []
# #     for x_true, y_true in test_generator:
# #         y_pred = model.predict(x_true).argmax(axis=1) - 1
# #         # y_true = y_true[:, 0]
#         pred.extend(y_pred)
df = pd.DataFrame()
test_result_pd = pd.read_csv(data_dir + 'processed_data/new_test_df.csv')
test_result_pd['微博id'] = test_result_pd['微博id'].apply(lambda x: str(x))
df['id'] = test_result_pd['微博id'].tolist()
df['y'] = sub
df.to_csv('keras_test_result.csv', index=False, encoding='utf-8')
