python3 run_classifier.py \
--data_dir=data/bert_classifier_training \
--bert_model=bert-base-uncased \
--task_name=yelp \
--output_dir=models \
--max_seq_length=70 \
--do_train \
--do_lower_case \
--train_batch_size=32 \
--num_train_epochs=1 