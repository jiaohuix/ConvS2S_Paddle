# Mlp-Mixer论文复现

​		convs2s是基于卷积的翻译模型。

## Tree

```
# 目录结构
├── LICENSE
├── README.md
├── align.py # 对齐
├── bleu.sh # 评估
├── ckpt # 权重
├── config #配置
│   ├── base.yaml
│   ├── en2de.yaml #英德
│   └── en2ro.yaml
├── data
│   ├── __init__.py
│   ├── data.py # 数据加载
│   └── sampler.py # 采样器
├── eval.py # 训练中验证
├── logs # 日志
├── loss # 损失函数，速度太慢，弃置
├── main.py # 主函数
├── main_multi_gpu.py #多卡训练脚本
├── models #模型文件
├── predict.py # 生成预测翻译
├── requirements.txt
├── run_multi_gpu.sh # 多卡训练命令
└── wmt14ende_newstest #数据集
```

## Train

```
单卡
python main.py --config conf/en2de.yaml --mode train
多卡
bash run_multi_gpu.sh
```

## Evaluate

```
# 先生成预测文件
python main.py  --config conf/en2de.yaml --mode pred
# 再用bleu.sh评估
bash bleu.sh
```

## Link

- 百度网盘：链接：-  提取码：-
- aistudio：https://aistudio.baidu.com/aistudio/projectdetail/2328014

