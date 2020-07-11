#!/usr/bin/env python
# coding: utf-8
###分析特征情况
# In[29]:


import pickle 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from keras.utils import np_utils


with open('/home/none404/hm/lei/CCF_emotion/ours_vector.pickle','rb') as f1:
    ours_vector=pickle.load(f1)
ours_vector = np.array(ours_vector)


# In[54]:


import pandas as pd
test= pd.read_csv('/home/none404/hm/lei/CCF_emotion/result_8_4_16.csv',)


# In[55]:


ours_vector.shape
hidden_features=ours_vector


# In[56]:


#-------------------------------PCA,tSNE降维分析--------------------------------
pca = PCA(n_components=3)# 总的类别
pca_result = pca.fit_transform(hidden_features)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

#Run T-SNE on the PCA features.
tsne = TSNE(n_components=2, verbose = 1)
tsne_results = tsne.fit_transform(pca_result[:5000])
#-------------------------------可视化--------------------------------
color_map = np.array(test['y'][:5000])
plt.figure(figsize=(10,10))
for cl in range(3):# 总的类别
    indices = np.where(color_map==cl-1)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl-1)
plt.legend()
plt.show()


# In[48]:





# In[ ]:





# In[52]:


with open('/home/none404/hm/lei/CCF_emotion/ori_vector_list.pickle','rb') as f1:
    ori_vector=pickle.load(f1)
ori_vector = np.array(ori_vector)
hidden_features =ori_vector

#-------------------------------PCA,tSNE降维分析--------------------------------
pca = PCA(n_components=3)# 总的类别
pca_result = pca.fit_transform(hidden_features)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

#Run T-SNE on the PCA features.
tsne = TSNE(n_components=2, verbose = 1)
tsne_results = tsne.fit_transform(pca_result[:5000])
#-------------------------------可视化--------------------------------
color_map = np.array(test['y'][:5000])
plt.figure(figsize=(10,10))
for cl in range(3):# 总的类别
    indices = np.where(color_map==cl-1)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl-1)
plt.legend()
plt.show()


# In[ ]:




