import argparse
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, load_dataset
import os
import accelerate
from accelerate.state import PartialState
from accelerate.state import AcceleratorState

#AcceleratorState._reset_state()  # 🚀 강제로 기존 상태 리셋

#accelerator = accelerate.Accelerator(cpu=False, mixed_precision="fp16", device_placement=True)


# 명령줄 인수를 처리하기 위한 argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, required=True, help="경로에 있는 데이터셋")
    parser.add_argument('--output_path', type=str, required=True, help="모델 출력 경로")
    return parser.parse_args()

def set_environment_variables():
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def train_model(args):
    
    set_environment_variables()
    
    # print("⚠️Distributed Type:", accelerator.state.distributed_type)
    # print("⚠️Process Index:", accelerator.state.process_index)
    # print("⚠️Local Process Index:", accelerator.state.local_process_index)

    BASE_MODEL = "beomi/Llama-3-Open-Ko-8B"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = '<|reserved_special_token_0|>'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # device_index = accelerator.process_index
    # device_map = {"": device_index}
    #device_map="auto", 
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        #device_map={"":accelerate.Accelerator().local_process_index}
    )

    lora_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    dataset = load_dataset("json", data_files=args.input_dataset, split="train")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_path,
        deepspeed="ds_config.json",
        optim="paged_adamw_32bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=1000,
        learning_rate=2e-4,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        no_cuda=False,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # if accelerator.is_main_process:
    #     model.save_pretrained(args.output_path)

if __name__ == "__main__":
    # print('`accelerate.launch` 실행 여부 확인')
    # state = PartialState()
    # print(f"🚀 Distributed Type: {state.distributed_type}")
    # print(f"🔢 Process Index: {state.process_index}")
    # print(f"🖥️ Local Process Index: {state.local_process_index}")

    args = parse_args()
    train_model(args)