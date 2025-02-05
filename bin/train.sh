deepspeed --master_port=29505 --include="localhost:0,1,2,3,4,5,6,7" supervised_finetuning.py \
	--train_type dynamic-shot \
	--aicl_schema all \
	--windows_size 10 \
	--output_dir ../output/llama_a2g11_dynamic_windows10 \
	--model_type llama \
	--model_name_or_path ../models/llama2-hf-chat-7B/ \
	--tokenizer_name_or_path ../models/llama2-hf-chat-7B/ \
	--cache_dir ../output/llama_a2g11_dynamic_windows10
