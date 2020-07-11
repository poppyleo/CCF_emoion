各个文件的文件路径，模型路径，相关参数配置在config.py中
**执行步骤**
```python preprocess.py```：预处理数据
```python train_fine_tune```：把数据通过utils处理成bert所接受的形式，再输入至模型进行微调
```python postprocess/check.F1```：找到训练好的模型里，线下评分最高的那个
```python get_bestweight.py```:找到样本不均衡的动态权重
```python predict.py```：预测，生成结果文件以及保存概率
```python ensemble/ensemble.py```：投票融合&&概率融合
**Tips**
1、注意配置文件和代码绝对路径与相对路径
2、框架中utils.py和model.py为不同任务需要修改的地方  
