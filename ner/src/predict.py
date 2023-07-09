#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   predict.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   predict入口
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import paddle

import dygraph
from data_loader import DataLoader
from label_encoder import LabelEncoder
from models.models import BIGRU_CRF,BILSTM_CRF,LSTM_CRF,GRU_CRF
from paddlenlp.embeddings import TokenEmbedding,list_embedding_name

class Predict:
    """模型预测
    """
    def __init__(self, predict_conf_path):
        self.predict_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.predict_conf.read(predict_conf_path)

        self.label_encoder = LabelEncoder(label_id_info=self.predict_conf["DATA"]["label_encoder_path"],
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))
        self.vocab_tokenizer = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                              extended_vocab_path=self.predict_conf["DATA"]["vocab_path"],
                                              unknown_token="[UNK]"
                                              ).vocab

    def run(self):
        """执行入口
        """
        if self.predict_conf["RUN"].getboolean("train_or_predict"):

            label_encoder = self.label_encoder

            model_type = self.predict_conf["model"]["model_type"]
            assert model_type in ["bilstm+crf", "bigru+crf", "lstm+crf", "gru+crf"], \
                "model_type must in [bilstm+crf, bigru+crf, lstm+crf, gru+crf]"
            use_w2v_emb = self.predict_conf["model"].getboolean("use_w2v_emb")
            if model_type == "bigru+crf":
                model_predict = BIGRU_CRF(emb_size=300,
                                  hidden_size=256,
                                  num_classes=label_encoder.size() * 2 - 1,
                                  use_w2v_emb=use_w2v_emb,
                                  extended_vocab_path=self.predict_conf["DATA"]["vocab_path"]
                                  )
            elif model_type == "gru+crf":
                model_predict = GRU_CRF(emb_size=300,
                                hidden_size=256,
                                num_classes=label_encoder.size() * 2 - 1,
                                use_w2v_emb=use_w2v_emb,
                                extended_vocab_path=self.predict_conf["DATA"]["vocab_path"]
                                )
            elif model_type == "lstm+crf":
                model_predict = LSTM_CRF(emb_size=300,
                                hidden_size=256,
                                num_classes=label_encoder.size() * 2 - 1,
                                use_w2v_emb=use_w2v_emb,
                                extended_vocab_path=self.predict_conf["DATA"]["vocab_path"]
                                )
            else:
                model_predict = BILSTM_CRF(emb_size=300,
                                   hidden_size=256,
                                   num_classes=label_encoder.size() * 2 - 1,
                                   use_w2v_emb=use_w2v_emb,
                                   extended_vocab_path=self.predict_conf["DATA"]["vocab_path"]
                                   )

            dygraph.load_model(model_predict, self.predict_conf["MODEL_FILE"]["model_best_path"])

            predict_data = []
            text_list = []
            length_list = []
            origin_texts = []
            mark = 0
            tmp_d = {}
            for line in sys.stdin:
                mark = mark + 1
                cols = line.strip("\n").split("\t")
                origin_text = cols[0]
                text = self.vocab_tokenizer.to_indices(list(origin_text))
                text_list.append(text)
                length_list.append(len(text))
                origin_texts.append(origin_text)
                tmp_d[origin_text] = cols

                if mark == 32:
                    predict_data = list(zip(text_list, length_list))
                    pre_label, pre_entity = dygraph.predict(model=model_predict,
                                                            predict_data=predict_data,
                                                            label_encoder=label_encoder,
                                                            batch_size=self.predict_conf["model"].getint("batch_size"),
                                                            max_seq_len=self.predict_conf["model"].getint("max_seq_len"),
                                                            max_ensure=True,
                                                            with_label=False)
                    for origin_text, entity in zip(origin_texts, pre_entity):
                        print(origin_text,"\t", "\t".join(tmp_d[origin_text][1:]), "\t", json.dumps(entity, ensure_ascii=False))
                    predict_data = []
                    text_list = []
                    length_list = []
                    origin_texts = []
                    mark = 0
            if mark != 0:
                predict_data = list(zip(text_list, length_list))
                pre_label, pre_entity = dygraph.predict(model=model_predict,
                                                        predict_data=predict_data,
                                                        label_encoder=label_encoder,
                                                        batch_size=self.predict_conf["model"].getint("batch_size"),
                                                        max_seq_len=self.predict_conf["model"].getint("max_seq_len"),
                                                        max_ensure=True,
                                                        with_label=False)
                for origin_text, entity in zip(origin_texts, pre_entity):
                    print(origin_text, "\t", "\t".join(tmp_d[origin_text][1:]), "\t", json.dumps(entity, ensure_ascii=False))

if __name__ == "__main__":
    Predict(predict_conf_path=sys.argv[1]).run()
