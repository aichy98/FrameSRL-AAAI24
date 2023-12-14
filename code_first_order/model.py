from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Optional, Tuple, List
from transformers.modeling_outputs import ModelOutput
import allennlp.modules.span_extractors.max_pooling_span_extractor as max_pooling_span_extractor
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_log_softmax
from opt_einsum import contract

@dataclass
class FrameSRLModelOutput(ModelOutput):
    """

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    pred_logits: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        #hidden_size = input_size
        #self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, input_size)
        #self.fc3 = nn.Linear(input_size, output_size)
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        #x1 = F.relu(self.fc1(x))
        #x2 = F.relu(self.fc2(x1) + x)  # Add residual connection
        #output = self.fc3(x2)
        output = F.relu(self.fc1(x))
        return output

class BertForFrameSRL(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.FE_extractor = max_pooling_span_extractor.MaxPoolingSpanExtractor(config.hidden_size)
        self.weight1=nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
        self.weight2=nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
        self.mlp_start=MLP(config.hidden_size, config.hidden_size)
        self.mlp_end=MLP(config.hidden_size, config.hidden_size)
        self.mlp_FE=MLP(config.hidden_size, config.hidden_size)
        self.criterion=nn.CrossEntropyLoss()
        #self.max_length=20
        #self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        #self.loss_fct_nll = nn.NLLLoss(ignore_index=-1)
        self.post_init()
        self.weight1.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.weight2.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        FE_token_idx: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        context_length: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]

        pred_logits=[]

        total_loss = torch.tensor(0., requires_grad=True).to(sequence_output.device)

        for i in range(input_ids.shape[0]):
            sequence_output_i=sequence_output[i]
            row_sums = torch.sum(FE_token_idx[i], dim=1)
            non_zero_rows = row_sums != 0
            FE_num = non_zero_rows.sum()
            FE_embeddings=self.FE_extractor(sequence_output_i.unsqueeze(0), FE_token_idx[i,:FE_num].unsqueeze(0)).squeeze(0)
            cont_len=context_length[i].item()
            cont_emb=sequence_output_i[:cont_len]
            FE_embeddings=self.mlp_FE(FE_embeddings)
            start_emb=self.mlp_start(cont_emb)
            end_emb=self.mlp_end(cont_emb)
            # start_logit=torch.einsum('xi,ij,yj->xy', FE_embeddings, self.weight1, start_emb)
            # end_logit=torch.einsum('xi,ij,yj->xy', FE_embeddings, self.weight2, end_emb)
            start_logit=contract('xi,ij,yj->xy', FE_embeddings, self.weight1, start_emb, backend='torch')
            end_logit=contract('xi,ij,yj->xy', FE_embeddings, self.weight2, end_emb, backend='torch')
            start_logit[:,0]=0
            end_logit[:,0]=0

            # loss1 = self.criterion(start_logit, start_positions[i, :FE_num])
            # loss2 = self.criterion(end_logit, end_positions[i, :FE_num])
            # total_loss += (loss1 + loss2) / 2
            expanded_start_logit=start_logit.unsqueeze(2)  # shape becomes (FE, num_span, 1)
            expanded_end_logit=end_logit.unsqueeze(1)  # shape becomes (FE, 1, num_span)
            added_logit=expanded_start_logit+expanded_end_logit  # shape is (FE, num_span, num_span)
            j=torch.arange(cont_len).unsqueeze(1).to(sequence_output.device)
            k=torch.arange(cont_len).unsqueeze(0).to(sequence_output.device)
            mask = (j<=k)#&(k-j<self.max_length)  # shape is (num_span, num_span)
            mask[0, 1:]=False
            added_logit[:, ~mask] = float('-inf')  # set entries where j > k to negative infinity
            flattened_logit=added_logit.view(FE_num, -1)  # shape is (FE, num_span^2)
            pred_logits.append(flattened_logit)

            gold_positions=torch.stack((start_positions[i, :FE_num], end_positions[i, :FE_num]), dim=-1)
            #mask_gold = (gold_positions[:, 1]-gold_positions[:, 0]) >= self.max_length
            #gold_positions[mask_gold] = torch.tensor([0, 0]).to(gold_positions.device)
            gold_position_indices=gold_positions[:, 0]*cont_len+gold_positions[:, 1]  # shape is (FE)
            total_loss += self.criterion(flattened_logit,gold_position_indices)


        if not return_dict:
            output = pred_logits + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FrameSRLModelOutput(
            loss=total_loss,
            pred_logits=pred_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        



