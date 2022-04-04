# ShufflingBias
Here is the implementation of our paper Overcoming Language Priors in Visual Question Answering by Shuffling Bias with Two-stageTraining Strategy
This repository contains code modified from [SSL](https://github.com/CrossmodalGroup/SSL-VQA), many thanks! 
Most of the code content is based on the above link, the specific changes are as follows:

## Requirements
python 3.8.8
pytorch 1.9.0+cu111
cuda 11.0
gpu nvidia2080ti（11G）


## data
We uploaad the relevant data to the [baidu disk] 链接：https://pan.baidu.com/s/1efOcEBtR5CZtAQ8JyIRriA 提取码：lh5w，which include best model pth

## code
This file contains the complete modified code

## training

```python
CUDA_VISIBLE_DEVICES=3 python main.py --dataroot data/vqacp2/ --img_root data/coco/ --output [0.1-2]/ --self_loss_weight 3 --ml_loss
```

Note that，we have supply some shuffling bias in cache file。For the get_bias function we apply it to the training set, not to the test set
