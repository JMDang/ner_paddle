#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_2fc.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   ernie_2fc
"""

import sys
import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel

class Ernie2FcForTokenClassification(ErniePretrainedModel):
    r"""
    ERNIE Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        ernie (`ErnieModel`): 
            An instance of `ErnieModel`.
        num_classes (int, optional): 
            The number of classes. Defaults to `2`.
        dropout (float, optional): 
            The dropout probability for output of ERNIE. 
            If None, use the same value as `hidden_dropout_prob` 
            of `ErnieModel` instance `ernie`. Defaults to `None`.
    """

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(Ernie2FcForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        # self.classifier = nn.Linear(self.ernie.config["hidden_size"],
        #                             num_classes)

        self.classifiers = paddle.nn.LayerList(
            [nn.Linear(self.ernie.config["hidden_size"], num_classes) for _ in range(2)]
        )

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import Ernie2FcForTokenClassification, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = Ernie2FcForTokenClassification.from_pretrained('ernie-1.0')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits_list = [paddle.unsqueeze(layer(sequence_output), axis=1) for layer in self.classifiers]
        logits = paddle.concat(logits_list, axis=1)

        return logits

if __name__ == "__main__":
    ernie = Ernie2FcForTokenClassification.from_pretrained("ernie-1.0",num_classes=417)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    logits = ernie(**inputs)
    print(logits)


