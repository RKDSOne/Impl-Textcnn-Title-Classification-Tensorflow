# coding=utf-8
import sys
from pyltp import *
# from optOnMysql.NewsOnMysql import *
import re
import string
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
print(BasePath)
class MySegment(object):
    def __init__(self):
        self.model = BasePath + '/ltp_data/cws.model'
        self.lexicon = 'lexi.model'
        self.segmentor = Segmentor()
        self.segmentor.load(self.model)

    def load_default_model(self):
        self.segmentor.load(self.model)

    def sen2word(self,sen):
        # 测试:出现句子为单字
        # print(sen)
        
        nonsentence = self.remove_punctuation(sen.decode('utf8'))
        word_obj = self.segmentor.segment(nonsentence)

        word_list = list(word_obj)
        # print("----------")
        # print(word_list)
        return word_list

    def senlist2word(self,sentence_list):
        '''
            input:
                sentent the sen to be seg
            output:
                the word list
        '''
        word_list = list()
        # print(1)
        for sentence in sentence_list:
            # print(sentence)
            word_obj = self.sen2word(sentence)
            word_list = word_list + word_obj
        return word_list

    def paraph2sen(self,paraph):
        # print(paraph)
        sentence_obj = SentenceSplitter.split(paraph.encode('utf8'))
        sentence_list = list(sentence_obj)
        print(sentence_list)
        return list(sentence_list)

    def remove_punctuation(self,sentence):
        return ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence)).encode('utf8')
    def close(self):
        self.segmentor.release()
        print(self.segmentor.release())

if __name__ == "__main__":
    print("segment")
    # Segmentor = MySegment()
    # Segmentor.paraph2sen()
