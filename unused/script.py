import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, get_peft_model

def generate_prompt(example):
    dialog = example['dialog']
    summarization = example['summarization']
    prompt = "<|begin_of_text|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n아래 문장을 요약해주세요\n\n{dialog}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{summarization}<|end_of_text|>"
    example['prompt'] = prompt
    return example

BASE_MODEL = "beomi/Llama-3-Open-Ko-8B"#"meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'
tokenizer.pad_token = '<|reserved_special_token_0|>'


dataset = load_dataset("csv", data_files="gs://bun-bucket-test1/data/final_v12.2.1.csv", split="train")
transformed_dataset = dataset.map(generate_prompt)
dataset = transformed_dataset.map(lambda samples: tokenizer(samples["prompt"], add_special_tokens=False, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"), batched=True)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# from accelerate import Accelerator
# device_index = Accelerator().process_index
# device_map = {"": device_index}

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map = "auto", quantization_config=bnb_config)
training_args = TrainingArguments(
        output_dir=".bun-bucket-test1/outputs/llama3-8b/v12.2.1",
        optim="paged_adamw_32bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=1000,
        learning_rate=2e-4,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        fp16=True,
        save_strategy="steps",
        save_steps=1000,
        max_steps=5000,
        warmup_ratio=0.03,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        remove_unused_columns=True,
        report_to=None
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

import transformers
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()