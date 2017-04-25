# Impl-Textcnn-Title-Classification-Tensorflow
This is a Implementation of TextCNN which come from https://github.com/dennybritz/cnn-text-classification-tf.
  
This code is used for classifying News' Titles from Sina News.

this code is a baseline version of random word embedding, we also use three method to improve the precision：
1.  pretrain word2vec from our corpus.
2.  average pooling for word embedding in a title.
3.  maxmin pooling for word embedding in a title.
