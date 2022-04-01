import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
from bias_loss_f import *
from dataset_vqacp import Dictionary, VQAFeatureDataset
from collections import defaultdict, Counter
from base_model import Model
import utils
import opts
from train import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)

def get_bias(train_dset,eval_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["anno"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["anno"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    model = Model(opt)
    model.debias_loss_fn = LearnedMixin(0.36)
    model = model.cuda()
    model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)
    get_bias(train_dset, eval_dset)
    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils.trim_collate)
    opt.use_all = 1
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils.trim_collate)

    train(model, train_loader, eval_loader, opt)
