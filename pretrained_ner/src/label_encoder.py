#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   label_encoder.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   label_encoder
"""
import sys

class LabelEncoder(object):
    """类别处理工具
    """
    def __init__(self, label_id_info, isFile=True):
        """初始化类别编码类
        [in]  label_id_info: str/dict, 类别及其对应id的信息
              isFile: bool, 说明信息是字典还是文件，若是文件则从其中加载信息
        """
        if isFile:
            self.label_id_dict = dict()
            with open(label_id_info, "r") as rf:
                for line in rf:
                    parts = line.strip("\n").split("\t")
                    self.label_id_dict[parts[1]] = int(parts[0])
        else:
            if isinstance(label_id_info, dict):
                 self.label_id_dict = label_id_info
            else:
                raise ValueError("unknown label_id_info type: {}".format(type(label_id_info)))

        self.id_label_dict = {v: k for k, v in self.label_id_dict.items()}

    def transform(self, label_name):
        """类别名称转id
        [in]  label_name: str, 类别名称
        [out] label_id: id, 类别名称对应的id
        """
        if label_name not in self.label_id_dict:
            raise ValueError("unknown label name: %s" % label_name)
        return self.label_id_dict[label_name]

    def inverse_transform(self, label_id):
        """id转类别名称
        [in]  label_id: id, 类别名称对应的id
        [out] label_name: str, 类别名称
        """
        if label_id not in self.id_label_dict:
            raise ValueError("unknown label id: %s" % label_id)
        return self.id_label_dict[label_id]

    def size(self):
        """返回类别数
        [out] label_num: int, 类别树目
        """
        return len(set(self.label_id_dict.keys()))

    def labels(self):
        """返回全部类别 并固定顺序
        """
        return sorted(set(self.label_id_dict.keys()))

if __name__ == "__main__":
    label_encoder = LabelEncoder("../input/label.txt")
    print(label_encoder.label_id_dict)
