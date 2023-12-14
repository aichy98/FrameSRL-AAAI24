python run.py \
--model_name_or_path bert-base-uncased \
--output_dir /storage/data/aichy/model15_unary_0_5e-5_42/ \
--train_file ../data15/train_instance_dic_prompt.npy \
--validation_file ../data15/dev_instance_dic_prompt.npy \
--test_file ../data15/test_instance_dic_prompt.npy \
--num_train_epochs 20 \
--save_best \
--do_predict \
--num_warmup_steps 0.05 \
--log_every_step 20 \
--max_grad_norm 5.0 \
--seed 42 \
--learning_rate 5e-5 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 