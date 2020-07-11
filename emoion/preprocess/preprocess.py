import pandas as pd
import re
from config import Config
import os
from sklearn.model_selection import KFold
config = Config()
data_dir = config.data_process
print(data_dir)

# 原始数据集
train_df = pd.read_csv(os.path.join(data_dir, 'train/nCoV_100k_train.labled.csv'), encoding='utf-8-sig')
test_df = pd.read_csv(os.path.join(data_dir, 'test/nCov_10k_test.csv'), encoding='utf-8-sig')


# 找到训练集测试集所有的非中文英文数字符号
additional_chars = set()
for t in list(test_df['微博中文内容']) + list(train_df['微博中文内容']):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))
print('文中出现的非中英文的数字符号：', additional_chars)

# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
print('保留的标点:', extra_chars)
additional_chars = additional_chars.difference(extra_chars)


def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")  # 过滤www开头的网址
    x = re.sub('\s', '', x)   # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    # 删除奇怪标点
    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x


train_df['text'] = train_df['微博中文内容'].fillna('')
test_df['text'] = test_df['微博中文内容'].fillna('')
train_df['label'] = train_df['情感倾向'].fillna('')


train_df['text'] = train_df['text'].apply(stop_words)
test_df['text'] = test_df['text'].apply(stop_words)
test_df['cur_text_len'] = [i.__len__() for i in test_df['text']]
print(test_df['cur_text_len'].describe())
"""
修复错误标签  && 打标
"""
idx_list= [35168, 46324, 46325, 46326, 46327, 46328, 46329, 46330, 46331,
            46332, 46333, 46334, 46335, 46336, 46337, 46338, 46339, 46340,
            46341, 46342, 46343, 46344, 46345, 46346, 46347, 46348, 46349,
            46350, 46351, 46352, 46353, 46354, 46355, 46356, 46357, 46358,
            46359, 46360, 46361, 46362, 46363, 46364, 46365, 46366, 46367,
            46368, 46369, 46370, 46371, 46372, 46373, 46374, 46375, 46376,
            46377, 46378, 46379, 46380, 46381, 46382, 46383, 46384, 46385,
            46386, 46387, 46388, 46389, 46390, 46391, 46392, 46393, 46394,
            46395, 46396, 46397, 47401, 48178, 48507, 48544, 49534, 49819,
             4439, 42034, 47474, 11068, 36583,  3520]
repair_label_list=[0,0,-1,-1,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,-1,1,-1,1,0,0,0,0,0,1,1,0,1,0,-1,0,-1,-1,0,0,0,0,0,0,0,
                   0,1,0,0,1,-1,1,0,0,0,0,0,0,-1,0,1,-1,0,0,1,1,1,1,1,-1,-1,0,-1]

for i, idx in enumerate(idx_list):
    train_df.loc[idx, 'label'] = repair_label_list[i]

"""
删除没有没有label的数据，（经查看和疫情无关）
"""
train_df.drop(list(train_df[train_df['label'] == ''].index), inplace=True)

"""
删除训练集没有text的数据，（经查看和疫情无关）
"""
train_df.drop(list(train_df[train_df['text'] == ''].index), inplace=True)

# 切分训练集，分成训练集和验证集
print('Train Set Size:', train_df.shape)
new_dev_df = train_df[80000:]  # 验证集  5: dev bert[3000:  4000] train dev   6: dev bert[:5000] dev dev  7:rober[:1000]
frames = [train_df[:40000], train_df[40001:80000]]
new_train_df = pd.concat(frames)  # 训练集
new_train_df = new_train_df.fillna('')

new_test_df = test_df[:]  # 测试集
print(len(new_test_df))
y_true = [0 for i in range(len(new_test_df))]  # 让测试集的标签全部为0
new_test_df['label'] = y_true

print('New Train Set Size:', new_train_df.shape)
print('New Dev Set Size:', new_dev_df.shape)

new_train_df.to_csv(data_dir + 'processed_data/new_train_df.csv', encoding='utf-8', index=False)
new_dev_df.to_csv(data_dir + 'processed_data/new_dev_df.csv', encoding='utf-8', index=False)
new_test_df.to_csv(data_dir + 'processed_data/new_test_df.csv', encoding='utf-8', index=False)

# """
# 对数据本地进行K折划分,保存至本地
# """
# kfold_data_path = config.data_process + 'processed_data/'
# kf = KFold(n_splits=5, shuffle=False)
# i = 1
# for train_index, val_index in kf.split(train_df['label']):
#     train = train_df.iloc[train_index]
#     val = train_df.iloc[val_index]
#     save_path = kfold_data+'fold'+str(i)+'/'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     train.to_csv(save_path+'train.csv', index=False, encoding='utf_8_sig')
#     val.to_csv(save_path + 'val.csv', index=False, encoding='utf_8_sig')
#     print(i)
#     i += 1
#
#
# """
# 保存测试集
# """
# test_df.to_csv(kfold_data_path + 'test.csv',index=False, encoding='utf_8_sig')


