nohup python inference.py --gpus 1 \
    --model_type llama \
    --template_name llama2 \
    --base_model ../output/llama_a2g11_zero/checkpoint-428/ \
    --tokenizer_path ../models/llama2-hf-chat-7B/ \
    --data_file ../data/icl_8k/in_context_learning_20240805_test.jsonl.arc \
    --output_file ../data/icl_8k/in_context_learning_20240805_test.jsonl.arc.a2g11.zero
