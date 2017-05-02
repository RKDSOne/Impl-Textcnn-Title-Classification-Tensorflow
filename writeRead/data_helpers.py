# coding=utf-8
import os
import sys
# reload(sys)
import json
import gensim
import cPickle
import logging
import itertools
import numpy as np
from Segment.MySegment import *
from writeRead.WriteRead import *
BasePath = sys.path[0]
# sys.setdefaultencoding('utf8')
from collections import Counter
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_json_data(userdir):
    readf = open(userdir,'r')
    json_data = readf.read()
    readf.close()
    decode_json = json.loads(json_data)
    return decode_json

def save2json(userdir,data2save):
    encode_json = json.dumps(data2save)
    writef = open(userdir,'w')
    writef.write(encode_json)
    writef.close()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def One_Hot_Encoding(cate_list):
    '''
        input:
            cate_list: title对应类别list
        output:
            label_reform: One_Hot编码后的cate_list
    '''
    cate_dict = dict()
    cate_set = set(cate_list)
    i = 0
    for label in cate_set:
        print(label)
        print(i)
        cate_dict[label] = i
        i += 1
    print("_______________________________________________")
    save2json(BasePath + "/data/cate_list.json",cate_dict)
    # one-hot encoing
    labels = np.array([cate_dict[tmp] for tmp in cate_list])
    label_reform = (np.arange(len(cate_dict)) == labels[:,None]).astype(np.float32)
    return label_reform

def segment(title_list):
    '''
        input:
            title_list: 标题list
        output:
            seg_title_list: 分好词的标题list,以空格间隔
    '''
    myseg = MySegment()
    seg_title_list = list()
    # print(title_list)
    for title in title_list:
        # print(title)
        #更改 senlist2word改为sen2word
        title_wordlist = myseg.sen2word(title.decode('utf-8'))
        # print(title_wordlist)
        seg_title_list.append(' '.join(title_wordlist))
    return seg_title_list
# def load_data_and_labels():

def get_dev_train_data(file_path):
    '''
        input:
            file_path: courpsus训练数据所在目录
        output:
            seg_title: 分好词的标题list
            label_reform: one-hot编码后的类别
    '''

    title_list = list()
    cate_list = list()
    opt_file = WriteRead(file_path)
    sample = opt_file.get_data()
    split_sample = sample.split('\n')[:-1]
    print(split_sample[0])
    for one in split_sample:
        # print(one.split('\t')[0])
        title_list.append(one.split('\t')[0])
        cate_list.append(one.split('\t')[1])
    seg_title = segment(title_list)
    label_reform = One_Hot_Encoding(cate_list)
    return [seg_title,label_reform]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def load_vocab(sentences):
    vocab=[]
    for sentence in sentences:
        vocab.extend(sentence.split())
    vocab=set(vocab)
    return vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print(vocab_size,layer1_size)
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs



def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """

    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1

    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    return word_vecs

if __name__ == "__main__":
    x,y = get_dev_train_data(BasePath + '/data/corpus.txt')
    print(x[0])
