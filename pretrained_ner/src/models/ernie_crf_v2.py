#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_crf_v2.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   ernie_crf_v2
"""

import sys
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieForTokenClassification
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from data_loader import DataLoader
from label_encoder import LabelEncoder

import logging
logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class ErnieCrfV2ForTokenClassification(nn.Layer):
    def __init__(self, ernie, crf_lr=0.1):
        super().__init__()
        self.num_labels = ernie.num_classes
        self.ernie = ernie  # allow ernie to be config
        self.crf = LinearChainCrf(self.num_labels, crf_lr=crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(self,
                input_ids,
                true_lengths,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        logits = self.ernie(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids)

        _, prediction = self.viterbi_decoder(logits, true_lengths)
        return logits, prediction



if __name__ == "__main__":
    ernie = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=3)
    model = ErnieCrfV2ForTokenClassification(ernie)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    inputs["true_lengths"] = paddle.to_tensor([7])
    logits, prediction = model(**inputs)
    print(prediction)
