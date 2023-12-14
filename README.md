# Code-for-Frame-Semantic-Role-Labeling-Using-Arbitrary-Order-Conditional-Random-Fields

Source Code for "Frame Semantic Role Labeling Using Arbitrary-Order Conditional Random Fields" at AAAI-2024.

Our code is based on [AGED](https://github.com/Zce1112zslx/AGED), thanks for their great work!

## Data Download
The preprocessed FrameNet 1.5 and 1.7 data are the same as [AGED](https://github.com/Zce1112zslx/AGED).

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

## Training

# first-order w/o exemplar
python run.py --model_name_or_path bert-base-uncased --output_dir /storage/data/aichy/model15_unary_0_5e-5_42/ --train_file ../data15/train_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 20 --save_best --do_predict --num_warmup_steps 0.05 --log_every_step 20  --max_grad_norm 5.0  --seed 42  --learning_rate 5e-5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32

# first-order w/ exemplar
python run.py --model_name_or_path bert-base-uncased --output_dir /storage/data/aichy/model15_unary_0_pretrain_5e-5_42/ --train_file ../data15/exemplar_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 5 --do_predict --num_warmup_steps 0.01 --log_every_step 160  --max_grad_norm 5.0  --seed 42  --learning_rate 5e-5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32

python run.py --model_name_or_path /storage/data/aichy/model15_unary_0_pretrain_5e-5_42/ --output_dir /storage/data/aichy/model15_unary_0_finetune_2.5e-5_42/ --train_file ../data15/train_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 10 --save_best --do_predict --num_warmup_steps 0.05 --log_every_step 20  --max_grad_norm 5.0  --seed 42  --learning_rate 2.5e-5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32

# arbitrary-order w/o exemplar
python run.py --model_name_or_path bert-base-uncased --output_dir /storage/data/aichy/model15_cpd_0_5e-5_42/ --train_file ../data15/train_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 20 --save_best --do_predict --num_warmup_steps 0.05 --log_every_step 20  --max_grad_norm 5.0  --seed 42  --learning_rate 5e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --rank 512

# arbitrary-order w/ exemplar
python run.py --model_name_or_path bert-base-uncased --output_dir /storage/data/aichy/model15_cpd_0_pretrain_5e-5_42/ --train_file ../data15/exemplar_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 5 --do_predict --num_warmup_steps 0.01 --log_every_step 160  --max_grad_norm 5.0  --seed 42  --learning_rate 5e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16  --rank 512 

python run.py --model_name_or_path /storage/data/aichy/model15_cpd_0_pretrain_5e-5_42/ --output_dir /storage/data/aichy/model15_cpd_0_finetune_2.5e-5_42/ --train_file ../data15/train_instance_dic_prompt.npy --validation_file ../data15/dev_instance_dic_prompt.npy --test_file ../data15/test_instance_dic_prompt.npy  --num_train_epochs 10 --save_best --do_predict --num_warmup_steps 0.05 --log_every_step 20  --max_grad_norm 5.0  --seed 42  --learning_rate 2.5e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16   --rank 512 



