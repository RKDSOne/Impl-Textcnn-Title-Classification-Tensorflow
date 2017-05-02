# coding=utf-8

from textCNN.textcnnModel import *
from textCNN.textcnnTrain import *
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
'''
    输入：数据集（按用户比例分为训练集，测试集,迭代次数）
    input ,
    输出：训练好的模型(路径)，训练、测试集的精度 list
    list((step,)):
'''
def impl_train(sample,percentage,num_steps):
    '''
        input:
            sample: 输入数据 格式 "title \t label\n"  
            percentage: 测试集所占比例
            num_steps: 训练迭代次数
        output:
            model_path: 例如:
            accuracy_list: 例如:
    '''
    model_path,accuracy_list = textcnnTrain(sample,percentage,num_steps)
    return model_path,accuracy_list

if __name__ == "__main__":
    corpus_path = BasePath + "/data/train.txt"
    model_path,accuracy_list = impl_train(corpus_path, 0.1, 20000)
