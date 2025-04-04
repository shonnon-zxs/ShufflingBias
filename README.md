# ShufflingBias
Here is the implementation of our paper Overcoming Language Priors in Visual Question Answering by Shuffling Bias with Two-stageTraining Strategy](https://ieeexplore.ieee.org/abstract/document/10214540/) (Accepted by IEEE Access 2023).

> Recent research has revealed the notorious language prior problem in visual question answering (VQA) tasks based on visual-textual interaction, which indicates that well-developed VQA
 models rely on learning shortcuts from questions without fully considering visual evidence.To tackle this problem, most existing methods focus on decreasing the incentive to learn prior knowledge by adding
 a question-only branch and becoming complacent by mechanically improving accuracy. However, these methods over-correct positive biases useful for generalization, leading to the degradation of performance
 on the VQA v2 dataset when cumulating their methods into other VQA architecture. In this paper, wepropose arobust shuffling language bias (SLB) approach to explicitly balance the prediction distribution,
 hopefully alleviating the language prior by increasing training opportunities for VQA models.Experiment results demonstrate that our method is cumulative with data augmentation and large-scale pre-training
 VQA architectures and achieves competitive performance on both the in-domain benchmark VQA v2 and out-of-distribution benchmark VQA-CP v2.
> ![image](https://github.com/user-attachments/assets/70cb7b7f-faa4-4aa6-be1f-42e4e8af75d0)


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

** Note that，we have supply some shuffling bias in cache file。For the get_bias function we apply it to the training set, not to the test set.**

## 7. Citation

```bibtex
@article{zhao2023overcoming,
  title={Overcoming language priors via shuffling language bias for robust visual question answering},
  author={Zhao, Jia and Yu, Zhengtao and Zhang, X and Yang, Ying},
  journal={IEEE Access},
  volume={11},
  pages={85980--85989},
  year={2023},
  publisher={IEEE}
}
  ```

## Acknowledgments
Our code is based on [SSL]([https://github.com/cshizhe/VLN-DUET](https://github.com/CrossmodalGroup/SSL-VQA)) for extract view features. Thanks for their great works!
