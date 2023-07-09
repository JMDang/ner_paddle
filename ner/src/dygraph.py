#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   dygraph.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   包含训练循环,预测,评估函数,加载模型函数
"""

import os
import sys
import numpy as np
import logging
import time
import copy
import paddle
from data_loader import DataLoader
import helper

logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

def train(model,
          train_data,
          label_encoder,
          dev_data=None,
          epochs=5,
          learning_rate=0.001,
          batch_size=32,
          max_seq_len=300,
          max_ensure=True,
          model_save_path=None,
          best_model_save_path=None,
          print_step=15
        ):
    """动态图训练
    """
    logging.info("train model start")
    train_start_time = time.time()

    model.train()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # paddle.nn.functional.cross_entropy()
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,
                                      parameters=model.parameters(),


                                      )

    cur_train_step = 0
    best_f1 = 0.0
    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果
        np.random.shuffle(train_data)
        train_data_batch = DataLoader.gen_batch_data(train_data,
                                                     batch_size,
                                                     max_seq_len,
                                                     max_ensure,
                                                     with_label=True)
        for cur_train_data, cur_train_length, cur_train_label, cur_train_entity in train_data_batch:
            cur_train_step += 1
            # 训练数据转为tensor
            cur_train_data = paddle.to_tensor(cur_train_data)
            cur_train_length = paddle.to_tensor(cur_train_length)
            cur_train_label = paddle.to_tensor(cur_train_label)
            # 生成loss

            logits, logits_decode = model(cur_train_data, cur_train_length)
            loss = paddle.mean(model.crf_loss(logits, cur_train_length, cur_train_label))

            if cur_train_step % print_step == 0:
                speed = cur_train_step / (time.time() - train_start_time)
                logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), speed))
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 每轮保存模型
        if model_save_path:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            paddle.save(model.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

        if dev_data:
            precision, recall, f1 = eval(model=model,
                                         eval_data=dev_data,
                                         label_encoder=label_encoder,
                                         batch_size=batch_size,
                                         max_seq_len=max_seq_len,
                                         max_ensure=max_ensure)
            logging.info('eval epoch %d, pre %.5f rec %.5f f1 %.5f' % (cur_epoch, precision, recall, f1))

            if best_model_save_path and f1 > best_f1:
                # 如果优于最优acc 则保存为best模型
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                paddle.save(model.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_f1 = f1

    logging.info("train model cost time %.4fs" % (time.time() - train_start_time))

def predict(model,
            predict_data,
            label_encoder,
            batch_size=32,
            max_seq_len=300,
            max_ensure=True,
            with_label=False):
    """ 动态图预测
    [IN]  model:
          predict_data: list[(input1[, input2, ...])], 待预测数据
          label_encoder: 标签编码器
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
          max_ensure: 是否为固定长度
          with_label: 是否包含真是标签
    """

    pre_label = []
    pre_entity = []
    rea_label = []
    rea_entity = []
    if with_label:
        predict_data_batch = DataLoader.gen_batch_data(
            predict_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_ensure=max_ensure,
            with_label=True)
        model.eval()
        for cur_predict_data, cur_predict_length, cur_rea_label, cur_rea_entity in predict_data_batch:
            cur_predict_data = paddle.to_tensor(cur_predict_data)
            cur_predict_length = paddle.to_tensor(cur_predict_length)

            logits, logits_decode = model(cur_predict_data, cur_predict_length)
            cur_pre_label = logits_decode.numpy()
            pre_label.extend(cur_pre_label)
            pre_entity.extend(helper.extract_entity_crf(pre_label=cur_pre_label,
                                                       predict_length=cur_predict_length.numpy(),
                                                       label_encoder=label_encoder))

            rea_label.extend(cur_rea_label)
            rea_entity.extend(cur_rea_entity)
        model.train()
        return pre_label, pre_entity, rea_label, rea_entity

    else:
        predict_data_batch = DataLoader.gen_batch_data(
            predict_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_ensure=max_ensure,
            with_label=False)
        model.eval()
        for cur_predict_data, cur_predict_length in predict_data_batch:
            cur_predict_data = paddle.to_tensor(cur_predict_data)
            cur_predict_length = paddle.to_tensor(cur_predict_length)

            logits, logits_decode = model(cur_predict_data, cur_predict_length)
            cur_pre_label = logits_decode.numpy()
            pre_label.extend(cur_pre_label)
            pre_entity.extend(helper.extract_entity_crf(pre_label=cur_pre_label,
                                                        predict_length=cur_predict_length.numpy(),
                                                        label_encoder=label_encoder))
        model.train()
        return pre_label, pre_entity


def eval(model,
         eval_data,
         label_encoder,
         batch_size=32,
         max_seq_len=300,
         max_ensure=True):
    """ eval
    [IN]  model:
          eval_data: list[(input1[, input2, ...], label)], 训练数据
          label_encoder: LabelEncoder, 类别转化工具
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
    [OUT] acc: float, 评估结果
    """
    pre_label, pre_entity, rea_label, rea_entity = predict(model=model,
                                                           predict_data=eval_data,
                                                           label_encoder=label_encoder,
                                                           batch_size=batch_size,
                                                           max_seq_len=max_seq_len,
                                                           max_ensure=max_ensure,
                                                           with_label=True)
    
    TP = 0.0
    pre_all = 0.000001
    rea_all = 0.000001
    for it1, it2 in zip(pre_entity, rea_entity):
        pre_all = pre_all + len(set([tuple(x) for x in it1]))
        rea_all = rea_all + len(set([tuple(x) for x in it2]))
        TP = TP + len(set([tuple(x) for x in it1]) & set([tuple(x) for x in it2]))

    precision = TP / pre_all
    recall = TP / rea_all
    f1 = 2 * precision * recall / (precision + recall + 0.000001)

    return precision, recall, f1

def load_model(model, model_path):
    """ 加载模型
    [in] model: 已构造好的模型结构
         model_path: str, 模型地址
    """
    if os.path.exists(model_path):
        logging.info("load model from {}".format(model_path))
        start_time = time.time()
        sd = paddle.load(model_path)
        model.set_dict(sd)
        logging.info("cost time: %.4fs" % (time.time() - start_time))
    else:
        logging.info("cannot find model file: {}".format(model_path))

if __name__ == "__main__":
    pass


