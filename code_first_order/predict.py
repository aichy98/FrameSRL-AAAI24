from allennlp.nn.util import get_mask_from_sequence_lengths
import numpy as np
import torch
from typing import List
import pdb
def post_process_function_greedy_pt(start_logits: torch.Tensor,
                                end_logits: torch.Tensor, 
                                context_length: torch.Tensor,):
    # bs=start_logits.shape[0]
    # start_pred = []
    # end_pred = []
    # for i in range(bs):
    #     num_span=context_length[i].item()
    #     start_logit=start_logits[i,:num_span].transpose(-1, -2)
    #     end_logit=end_logits[i,:num_span].transpose(-1, -2)
    #     FE=start_logit.shape[0]
    #     expanded_start_logit = start_logit.unsqueeze(2)  # shape becomes (FE, num_span, 1)
    #     expanded_end_logit = end_logit.unsqueeze(1)  # shape becomes (FE, 1, num_span)
    #     added_logits = expanded_start_logit + expanded_end_logit  # shape is (FE, num_span, num_span)
    #     added_logits[:, 0, 1:] = float('-inf')
    #     j = torch.arange(num_span).unsqueeze(1)
    #     k = torch.arange(num_span).unsqueeze(0)
    #     mask = j <= k  # shape is (num_span, num_span)
    #     # Apply the mask to logits
    #     masked_logits = added_logits.clone()  # make a copy of logits
    #     masked_logits[:, ~mask] = float('-inf')  # set entries where j > k to negative infinity
    #     # Find max values and their indices
    #     max_values, max_indices = masked_logits.view(FE, -1).max(dim=1)  # max_values and max_indices have shape (FE)
    #     # Compute span start and end positions
    #     span_starts = max_indices // num_span
    #     span_ends = max_indices % num_span
    #     start_pred.append(span_starts.cpu().numpy().tolist())
    #     end_pred.append(span_ends.cpu().numpy().tolist())
    # return start_pred, end_pred

    max_len = int(start_logits.shape[-2]) # (B, L, F)
    fe_num = int(start_logits.shape[-1])

    # predict start positions
    context_length_mask = get_mask_from_sequence_lengths(context_length.squeeze(), max_len)
    start_logits_masked = start_logits.data.masked_fill_(~context_length_mask.unsqueeze(-1), -float('inf'))
    start_pred = torch.argmax(start_logits_masked, dim=-2)
    start_mask = get_mask_from_sequence_lengths(start_pred.flatten(), max_len).reshape(-1, fe_num, max_len).permute(0, 2, 1)
    end_mask = start_mask ^ (context_length_mask.repeat([1, fe_num]).reshape(-1, fe_num, max_len).permute(0, 2, 1)) # end >= start

    # predict end positions
    end_logits_masked = end_logits.data.masked_fill_(~end_mask, -float('inf'))
    end_pred_ = torch.argmax(end_logits_masked, dim=-2)
    neg_mask = (start_pred == 0) # if start == 0, set end to 0
    end_pred = neg_mask * torch.zeros_like(end_pred_) + (~neg_mask) * end_pred_

    return start_pred.cpu().numpy().tolist(), end_pred.cpu().numpy().tolist()

def post_process_function_greedy(pred_logits: List[torch.FloatTensor],
                                context_length: torch.Tensor,):
    bs=len(pred_logits)
    start_pred = []
    end_pred = []
    for i in range(bs):
        num_span=context_length[i].item()
        added_logits=pred_logits[i]
        FE=added_logits.shape[0]
        added_logits=added_logits.reshape(FE,num_span,num_span)  # shape is (FE, num_span, num_span)
        added_logits[:, 0, 1:] = float('-inf')
        j = torch.arange(num_span).unsqueeze(1)
        k = torch.arange(num_span).unsqueeze(0)
        mask = j <= k  # shape is (num_span, num_span)
        # Apply the mask to logits
        masked_logits = added_logits.clone()  # make a copy of logits
        masked_logits[:, ~mask] = float('-inf')  # set entries where j > k to negative infinity
        # Find max values and their indices
        max_values, max_indices = masked_logits.view(FE, -1).max(dim=1)  # max_values and max_indices have shape (FE)
        # Compute span start and end positions
        span_starts = max_indices // num_span
        span_ends = max_indices % num_span
        start_pred.append(span_starts.cpu().numpy().tolist())
        end_pred.append(span_ends.cpu().numpy().tolist())
    return start_pred, end_pred

def post_process_function_with_max_len(start_logits: torch.Tensor,
                                        end_logits: torch.Tensor,
                                        context_length: torch.Tensor,
                                        max_len: int):
    # naive
    start_pred = []
    end_pred = []
    bsz = int(start_logits.shape[0])
    fe_num = int(start_logits.shape[-1])
    for i in range(bsz):
        cl = int(context_length[i])
        start_pred_tensor = torch.LongTensor([0]).to(start_logits.device)
        end_pred_tensor = torch.LongTensor([0]).to(end_logits.device)
        best_score = start_logits[i][0] + end_logits[i][0]
        for start in range(1, cl):
            for end in range(start, min(cl, start+max_len)):
                score = start_logits[i][start] + end_logits[i][end]
                mask = score > best_score
                start_pred_tensor = mask * start + (~mask) * start_pred_tensor
                end_pred_tensor = mask * end + (~mask) * end_pred_tensor
                best_score = mask * score + (~mask) * best_score

        start_pred.append(start_pred_tensor.cpu().numpy().tolist())
        end_pred.append(end_pred_tensor.cpu().numpy().tolist())
    
    return start_pred, end_pred
            

    

def calculate_F1_metric(start_pred: List,
                        end_pred: List,
                        FE_num: torch.Tensor,
                        gt_FE_word_idx: torch.Tensor,
                        gt_start_positions: torch.Tensor, 
                        gt_end_positions: torch.Tensor,
                        word_ids: torch.Tensor,
                        FE_core_pts: torch.Tensor):
    
    start_pred_lst = start_pred
    end_pred_lst = end_pred


    bsz = gt_FE_word_idx.shape[0]
    fesz = gt_FE_word_idx.shape[-1]
    TP = 0
    FP = 0
    FN = 0

    # print(end_pred_lst)
    for i in range(bsz):
        fe_num = int(FE_num[i][0])
        start_pred_word_lst = [int(word_ids[i][int(tok)]) for tok in start_pred_lst[i][:fe_num]]
        end_pred_word_lst = [int(word_ids[i][int(tok)]) for tok in end_pred_lst[i][:fe_num]]

        tp = 0
        fn = 0
        fp = 0

        for j in range(fesz):
            if int(gt_FE_word_idx[i][j]) == -1:
                break
            fe_idx = int(gt_FE_word_idx[i][j])
            fe_st = int(gt_start_positions[i][j])
            fe_ed = int(gt_end_positions[i][j])

            if start_pred_word_lst[fe_idx] == fe_st and end_pred_word_lst[fe_idx] == fe_ed:
                tp += float(FE_core_pts[i][fe_idx])
            else:
                fn += float(FE_core_pts[i][fe_idx])

        tp_fp = 0

        for j, x in enumerate(start_pred_word_lst):
            if x != -1:
                tp_fp += float(FE_core_pts[i][j])

        fp = tp_fp - tp

        TP += tp
        FN += fn
        FP += fp

    return TP, FP, FN        

