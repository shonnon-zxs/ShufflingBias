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
