import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model import LoRALinearExpertsWrapper  # → model.py에서 가져옴
from google.cloud import bigquery
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#"meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# 1. Load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token  # LLaMA는 특별한 pad 토큰이 없어서 eos로 대체

project_id = "prod-ai-project"
client = bigquery.Client(project=project_id)
data_sql = "where data_split in ('train') and create_date = (select max(create_date) from webtoon_translation.sft_dataset)"

def instruct_structure(prompt, system_prompt="""You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
        input_text, output_text = prompt.split('### target')
        input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{output_text.strip()}<|eot_id|>"""
    
train_sql = f"""          
              select prompt
              from webtoon_translation.sft_dataset
              {data_sql}
              """
train_df = client.query(train_sql).result().to_dataframe()
train_df['text'] = train_df.prompt.map(lambda x: instruct_structure(x))
train_dataset = Dataset.from_pandas(train_df[['text']])

#sft trainer는 text를 input으로 넣어주면 -> 알아서 tokenize, masking 알아서 해줌.
def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['text'])):
            output_texts.append(example['text'][i])
        return output_texts

response_template_with_context = '<|start_header_id|>assistant<|end_header_id|>'
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# 2. config
lora_r = 64
lora_alpha = 16
lora_dropout = 0

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    # target_modules=["q_proj", "v_proj"],
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "gate_up_proj", "gate_proj", "up_proj", "router"],
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
)

# 3. model
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        trust_remote_code=True,
)

# 4. MoE + Attention + FFN → LoRA 래핑
for layer in model.model.layers:
    moe = layer.mlp.experts
    if not isinstance(moe, LoRALinearExpertsWrapper):
        layer.mlp.experts = LoRALinearExpertsWrapper(moe, r=64, lora_alpha=16)


training_args = SFTConfig(
    output_dir='outputs/llama4',
    num_train_epochs=3,
    logging_steps=100,
    save_steps=1000000000000, # skip saving
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    weight_decay=0.05,
    bf16=True,
    remove_unused_columns=True,
    run_name="sft_webtoon_250408_llama4_109b",
    report_to="wandb",
    ddp_find_unused_parameters=False, # RuntimeError: Expected to mark a variable ready only once.
)
#wandb로그인 해야함

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=False,
        args=training_args,
)

trainer.train()

output_dir = os.path.join('outputs/llama4',"final_checkpoint")
trainer.save_model(output_dir)
trainer.model.save_pretrained(output_dir)
