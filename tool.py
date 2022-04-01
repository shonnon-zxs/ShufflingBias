    # different lmh to get bias
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

    # for ds in [train_dset, eval_dset]:
    for ds in [train_dset]:
        for ex in ds.entries:
            q_type = ex["anno"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]
            
            
            
            
            
            # u-u
            if epoch < 12:
                logits_pos, _= model(q, v, None, None, False)
                bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                loss = bce_loss_pos
            else:
                logits_pos, logits_neg, _,_, _ = model(q, v, a, bias, True)
                bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                self_loss = compute_self_loss(logits_neg, a)
                loss = bce_loss_pos + 2.8 * self_loss

            # r-u
            if epoch < 12:
                logits_pos, _, bce_loss_pos= model(q, v, a, bias, False)
                # bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                loss = bce_loss_pos
            else:
                logits_pos, logits_neg, _,_, _ = model(q, v, a, bias, True)
                bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                self_loss = compute_self_loss(logits_neg, a)
                loss = bce_loss_pos + 2.8 * self_loss

            # u-r
            if epoch < 12:
                logits_pos, _= model(q, v, None, None, False)
                bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                loss = bce_loss_pos
            else if epoch < 24:
                logits_pos, logits_neg, _,_, _ = model(q, v, a, bias, True)
                bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                self_loss = compute_self_loss(logits_neg, a)
                loss = bce_loss_pos + 2.8 * self_loss
            else:
                logits_pos, logits_neg, _,_, bce_loss_pos = model(q, v, a, bias, True)
                self_loss = compute_self_loss(logits_neg, a)
                loss = bce_loss_pos + 2.8 * self_loss

            # r-r
            if epoch < 12:
                logits_pos, _, bce_loss_pos= model(q, v, a, bias, False)
                # bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                loss = bce_loss_pos
            else:
                logits_pos, logits_neg, _,_, bce_loss_pos = model(q, v, a, bias, True)
                self_loss = compute_self_loss(logits_neg, a)
                loss = bce_loss_pos + 2.8 * self_loss
