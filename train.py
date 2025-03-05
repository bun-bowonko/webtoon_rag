import os
from typing import NamedTuple
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Input, Output, Dataset, Model
from google.cloud import aiplatform
from datetime import datetime


    
@dsl.component(base_image="asia-northeast3-docker.pkg.dev/dev-ai-project-357507/bun-docker-repo/bun-docker:latest", packages_to_install=["gcsfs"])
def preprocess_data(input_csv: str, output_dataset: Output[Dataset]):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    BASE_MODEL = "beomi/Llama-3-Open-Ko-8B"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = '<|reserved_special_token_0|>'
    
    def generate_prompt(example):
        dialog = example['dialog']
        summarization = example['summarization']
        prompt = "<|begin_of_text|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n아래 문장을 요약해주세요\n\n{dialog}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{summarization}<|end_of_text|>"
        example['prompt'] = prompt
        return example
    
    dataset = load_dataset("csv", data_files=input_csv, split="train")
    transformed_dataset = dataset.map(generate_prompt)
    tokenized_dataset = transformed_dataset.map(
        lambda samples: tokenizer(
            samples["prompt"], add_special_tokens=False, max_length=1024,
            padding="max_length", truncation=True, return_tensors="pt"
        ),
        batched=True
    )
    
    tokenized_dataset.to_json(output_dataset.path) #json형식으로 저장
    #tokenized_dataset.save_to_disk(output_dataset.path)
    
@dsl.component(base_image="asia-northeast3-docker.pkg.dev/dev-ai-project-357507/bun-docker-repo/bun-gpu:latest")
def train_model(input_dataset: Input[Dataset], output_model: Output[Model]):
    import torch
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model
    from datasets import load_from_disk, load_dataset
    import os
    import accelerate
    
    def set_environment_variables():
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    set_environment_variables()

    accelerator = accelerate.Accelerator(cpu=False, mixed_precision="fp16")
    
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    
    BASE_MODEL = "beomi/Llama-3-Open-Ko-8B"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = '<|reserved_special_token_0|>'
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map = "auto", quantization_config=bnb_config)

    lora_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    
    #dataset = load_from_disk(input_dataset.path)
    dataset = load_dataset("json", data_files=input_dataset.path, split="train")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=".bun-bucket-test1/outputs/llama3-8b/v12.2.1",
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
    
    model, dataset, data_collator = accelerator.prepare(model, dataset, DataCollatorForLanguageModeling(tokenizer, mlm=False))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    if accelerator.is_main_process:
        model.save_pretrained(output_model.path)


    
if __name__ == "__main__":
    
    from google_cloud_pipeline_components.v1.custom_job import utils    
    create_from_component = utils.create_custom_training_job_op_from_component

    @dsl.pipeline(name="llama3-training-pipeline")
    def llama3_pipeline(input_csv: str):
        preprocess_task = preprocess_data(input_csv=input_csv)
        #Operation 객체, 어떤 작업을 실행할지 정의된 구성 요소
        train_op = create_from_component(
            train_model,#component 함수
            machine_type= "g2-standard-24",
            accelerator_type="NVIDIA_L4",
            accelerator_count=2,
        )
        train_task = train_op(input_dataset=preprocess_task.outputs["output_dataset"]) #Task 객체, 파이프라인 내에서 실제로 실행되는 작업
    
    from kfp.v2.compiler import Compiler
    import google.cloud.aiplatform as aip
    
    Compiler().compile(pipeline_func=llama3_pipeline, package_path="llama3_pipeline_spec.json")
    
    aip.init(project="dev-ai-project-357507", location="asia-northeast3")
    pipeline_job = aip.PipelineJob(#operation 객체를 실행하는 대상
        job_id=f"my-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}",
        display_name="llama3-training-pipeline",
        template_path="llama3_pipeline_spec.json",
        pipeline_root="gs://bun-bucket-test1/pipeline-root",
        parameter_values={"input_csv": "gs://bun-bucket-test1/data/final_v12.2.1.csv"},
        enable_caching=False, #디버깅 다 끝나고 사용
        labels={
            'costcenter': 'cost_cto',
            'owner' : 'bun'
        }
    )
    pipeline_job.run(sync=True)