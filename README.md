# Code-for-Frame-Semantic-Role-Labeling-Using-Arbitrary-Order-Conditional-Random-Fields

Source Code for "Frame Semantic Role Labeling Using Arbitrary-Order Conditional Random Fields" at AAAI-2024.

Our code is based on PL-marker, thanks for their great work!

## Data Download
The preprocessed FrameNet 1.5 and 1.7 data are the same as [AGED](https://github.com/Zce1112zslx/AGED)

## Hyper-parameter Settings

We show the hyper-parameter settings in the following table.
| Hyper-parameters                  | Value             |
|-----------------------------------|-------------------|
| Pretrained Language Model         | bert-base-uncased |
| BERT embedding dimension          | 768               |
| batch size                        | 32                |
| optimizer                         | BertAdam          |
| scheduler                         | linear warmup     |
| warmup ratio (train only)         | 0.05              |
| warmup ratio (pretrain)           | 0.01              |
| warmup ratio (fine-tune)          | 0.05              |
| learning rate (train only)        | 5e-5              |
| learning rate (pretrain)          | 5e-5              |
| learning rate (fine-tune)         | 2.5e-5            |
| gradient clipping                 | 5.0               |
| MLP layers                        | 1                 |
| MLP activation function           | ReLU              |
| MLP dimension                     | 768               |
| rank dimension                    | 512               |
| mean-field inference iterations   | 3                 |
| epoch num (train only)            | 20                |
| epoch num (pretrain)              | 5                 |
| epoch num (finetune)              | 10                |




