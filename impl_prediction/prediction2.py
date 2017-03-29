# coding=utf-8

import os
import sys
reload(sys)
import time
import csv
import datetime
import numpy as np
import tensorflow as tf
from textCNN.textcnnModel import *
from textCNN.textcnnTrain import *
from tensorflow.contrib import learn
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
def get_predict_one_data(x):
    x_ret = list()
    myseg = MySegment()
    title_wordlist = myseg.sen2word(x.decode('utf-8'))
    x_seg = ' '.join(title_wordlist)
    x_ret.append(x_seg)
    return x_ret

def predict(x,check_dir):
    '''
    input:
        x: 未分词的title,例如: x = "两岁萌娃捡垃圾照走红网络"
        y: 类别,可不赋值,赋值时可给出准确率
        batch: 设置batch的大小
        checkpoint_dir: checkpoint目录
        pred_train: 预测或者测试
    output:
        cate: 分类的类别, 数字化类别
    '''
    # Eval Parameters
    checkpoint_dir = check_dir
    eval_train = True
    batch_size = 64
    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False


    if eval_train:
        x_raw = get_predict_one_data(x)
        # print(x_raw)
    else:
        # x_raw1,y_test1 = get_predict_one_data("社保基金会拟将所持股权划转至境内委托投资管理")
        # x_raw2,y_test2 = get_predict_one_data("两岁萌娃捡垃圾照走红网络")
        # x_raw = x_raw1+x_raw2
        # y_test = [1, 0]
    # print(3)
    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    # predict
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            # Generate batches for one epoch
            batches = batch_iter(list(x_test), batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
        sess.close()

    return all_predictions[0]
