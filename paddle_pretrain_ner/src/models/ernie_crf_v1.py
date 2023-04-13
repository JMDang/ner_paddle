#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_crf_v1.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   ernie_crf_v1
"""

import sys
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from paddlenlp.metrics import ChunkEvaluator

class ErnieCrfForTokenClassification(ErniePretrainedModel):

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieCrfForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.crf = LinearChainCrf(self.num_classes, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, with_start_stop_tag=False)
        self.apply(self.init_weights)


    def forward(self,
                input_ids,
                true_lengths,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        _, prediction = self.viterbi_decoder(emission, true_lengths)
        return emission, prediction


if __name__ == "__main__":
    model = ErnieCrfForTokenClassification.from_pretrained("ernie-1.0",num_classes=3)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    inputs["true_lengths"] = paddle.to_tensor([7])
    emission, prediction = model(**inputs)
    print(prediction)


