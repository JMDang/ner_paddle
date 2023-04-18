# paddle_pretrain_ner

### 代码说明

paddle框架写的基于ERNIE的命名实体识别逻辑。主要包含**ernie+2fc**进行半指针半标注的ner识别(优点可以覆盖实体嵌套的情况)，**ernie+crf**进行经典利用crf做最后一层进行ner识别。默认采用ernie+crf的方式训练，采用**ernie+2fc**可在[paddle_pretrain_ner/config/train_conf.ini](https://github.com/JMDang/paddle_pretrain_ner/blob/main/paddle_pretrain_ner/config/train_conf.ini)设置use_crf = false。此外 差分学习率  动转静推理  等配置。具体使用那种模型在config中可以配置，一目了然。

### 运行步骤

1.**标签配置:**在[paddle_pretrain_ner/input/label.txt](https://github.com/JMDang/paddle_pretrain_ner/blob/main/paddle_pretrain_ner/input/label.txt)中进行ner的类型配置，无论双指针还是crf方式进行ner都需要根据此文件进行数据预处理。

2.**数据准备:**在[paddle_pretrain_ner/input/train_data/train.txt](https://github.com/JMDang/paddle_pretrain_ner/blob/main/paddle_pretrain_ner/input/train_data/train.txt)中按照demo格式放入待训练的数据，两列，第一列为需要ner的文本，第二列为列表，列表的每个元素是每个实体的相关信息。同理，可在dev_data和test_data增加验证和测试数据

3.**环境准备:**按照requirments.txt安装相应的包即可，修改[paddle_pretrain_ner/env.sh](https://github.com/JMDang/paddle_pretrain_ner/blob/main/paddle_pretrain_ner/env.sh)配置cuda位置和使用的gpu卡，默认0卡。然后终端执行 `source env.sh `

4.**训练模型：**`python3 src/train.py config/train_conf.ini`模型会保存在paddle_pretrain_ner/model/ernie_finetune(动态图模型)和paddle_pretrain_ner/model/static_ernie(静态图模型用于推理部署) 文件夹中(脚本自动创建文件夹)

5.**预测模型：**`cat input/test_data/test.txt | python src/predict.py config/train_conf.ini` 预测结果会直接打印到终端，可自行重定向到指定文件。

**其他:**如果遇到任何问题，可以给本人邮箱776039904@qq.com发邮件，看到都会回复。







