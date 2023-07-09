#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   models.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   models
"""

import sys
import paddle
import paddle.nn as nn
from paddlenlp.embeddings import TokenEmbedding,list_embedding_name
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
import paddlenlp

def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fin:
        for line in fin:
            key = line.strip("\n")
            vocab[key] = i
            i += 1
    return vocab


class BILSTM_CRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 num_classes,
                 word_num=30000,
                 use_w2v_emb=True,
                 extended_vocab_path=None):
        super(BILSTM_CRF, self).__init__()
        if use_w2v_emb:
            '''
            embedding_name:预训练的词向量,list_embedding_name()可看到所有paddlenlp提供的预训练词向量
            extended_vocab_path:可默认,也可用自定义vocab,只是改变id和word的映射关系,不会改变word对应的预训练的向量,id->word->embedings
                                使用默认的词汇,[PAD],[UNK]的idx根据预训练的词包而定
                                个人更喜欢自定义[PAD]->0,[UNK]->1,[NUM]->2....others的映射关系,这样padding的都是0
            '''
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=2, direction="bidirect")
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.crf = LinearChainCrf(num_classes, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, true_lengths):
        embs = self.word_emb(inputs)
        output, _ = self.lstm(embs)
        emission = self.fc(output)
        _, prediction = self.viterbi_decoder(emission, true_lengths)
        return emission, prediction

class BIGRU_CRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 num_classes,
                 word_num=30000,
                 use_w2v_emb=True,
                 extended_vocab_path=None):
        super(BIGRU_CRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=2, direction="bidirect")
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.crf = LinearChainCrf(num_classes, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, true_lengths):
        embs = self.word_emb(inputs)
        output, _ = self.gru(embs)
        emission = self.fc(output)
        _, prediction = self.viterbi_decoder(emission, true_lengths)
        return emission, prediction

class LSTM_CRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 num_classes,
                 word_num=30000,
                 use_w2v_emb=True,
                 extended_vocab_path=None):
        super(LSTM_CRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.LSTM(emb_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = LinearChainCrf(num_classes, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, true_lengths):
        embs = self.word_emb(inputs)
        output, _ = self.gru(embs)
        emission = self.fc(output)
        _, prediction = self.viterbi_decoder(emission, true_lengths)
        return emission, prediction

class GRU_CRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 num_classes,
                 word_num=30000,
                 use_w2v_emb=True,
                 extended_vocab_path=None):
        super(GRU_CRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = LinearChainCrf(num_classes, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, true_lengths):
        embs = self.word_emb(inputs)
        output, _ = self.gru(embs)
        emission = self.fc(output)
        _, prediction = self.viterbi_decoder(emission, true_lengths)
        return emission, prediction

if __name__ == "__main__":

    """import numpy as np
    emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                         #可默认,也可用自定义vocab,只是改变id和word的映射关系,不会改变word对应的预训练的向量,id->word->embedings
                         #使用默认的词汇,[PAD],[UNK]的idx根据预训练的词包而定
                         #个人更喜欢自定义[PAD]->0,[UNK]->1,[NUM]->2....others的映射关系,这样padding的都是0
                         extended_vocab_path="../../input/data/word.dic",
                         unknown_token="[UNK]")
    print(emb.vocab.to_indices(["党"]))
    print(emb.vocab.to_tokens([1170]))
    embedings = emb(paddle.to_tensor([1170]))
    print(embedings.shape)
    print(embedings.numpy()[0][:10])
    poem = np.load("w2v.baidu_encyclopedia.target.word-word.dim300", allow_pickle=True)
    for index, word in enumerate(poem["vocab"]):
        if word == "党":
            print(word, index)
            print(poem["embedding"][index][:10])
    # print(emb(paddle.to_tensor([1170, 1171])))"""


    model = BIGRU_CRF(emb_size=300,
                      hidden_size=256,
                      num_classes=3,
                      use_w2v_emb=False)

    emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                         extended_vocab_path="../../input/vocab.txt",
                         unknown_token="[UNK]")
    # ernie.crf.parameters()
    inputs = emb.vocab.to_indices(list("隔壁王老师在哪里!"))
    inputs = paddle.to_tensor([inputs])
    true_lengths = paddle.to_tensor([5])
    print(inputs)
    print(true_lengths)
    emission, prediction = model(inputs=inputs, true_lengths=true_lengths)
    print(prediction)


