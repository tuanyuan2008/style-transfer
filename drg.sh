python3 openai_gpt_delete_retrive_and_generate.py \
--model_name openai-gpt \
--do_train \
--do_eval \
--train_dataset data/processed_files/dre_model/train/all.txt \
--eval_dataset data/processed_files/dre_model/test/all.txt \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_seq_length 85 \
--output_dir drg_model_weights