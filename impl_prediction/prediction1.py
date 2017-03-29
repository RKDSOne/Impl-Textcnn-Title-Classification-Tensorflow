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
def get_predict_one_data(x,y=None):
    x_ret = list()
    y_ret = list()
    myseg = MySegment()
    # print(1)
    title_wordlist = myseg.sen2word(x.decode('utf-8'))
    # print(title_wordlist)
    x_seg = ' '.join(title_wordlist)
    x_ret.append(x_seg)
    if y is not None:
        y_ret.append(cate_dict[y])
    return [x_ret,y_ret]

def predict(x,y,check_dir):
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
    # Data Parameters

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", check_dir, "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    # print("\nParameters:")
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))
    # print("")
    # CHANGE THIS: Load data. Load your own data here


    if FLAGS.eval_train:
        x_raw, y_test = get_predict_one_data(x,y)
        # print(1)
    else:
        x_raw1,y_test1 = get_predict_one_data("社保基金会拟将所持股权划转至境内委托投资管理")
        x_raw2,y_test2 = get_predict_one_data("两岁萌娃捡垃圾照走红网络")
        x_raw = x_raw1+x_raw2
        y_test = [1, 0]

    # Map data into vocabulary

    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    print("\nEvaluating...\n")
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
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
            batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})

                all_predictions = np.concatenate([all_predictions, batch_predictions])
                print(all_predictions[0])
    # Print accuracy if y_test is defined
    # predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    # out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    # print("Saving evaluation to {0}".format(out_path))
    # with open(out_path, 'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)
    return [all_predictions]
