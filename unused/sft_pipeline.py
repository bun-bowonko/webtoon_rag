import kfp
from kfp.v2 import dsl
from typing import NamedTuple

@dsl.component(
    base_image="asia-northeast3-docker.pkg.dev/prod-ai-project/tmp-h100/llama3.1-base@sha256:8125b2d27a8382d644041b37610ca65f4fa7bf61d5461f5665b34e6f190ab966",
    packages_to_install=["transformers==4.51.1"],
    output_component_file='webtoon-sft.yaml'
)
def train(project_id: str = "prod-ai-project",
        # base_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-405B-Instruct",
        # data_sql: str = "where data_split in ('train')",
        base_model_name_or_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        data_sql: str = "where data_split in ('train') and create_date = (select max(create_date) from webtoon_translation.sft_dataset)",
        max_seq_length: int = 8192,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        lora_r: int = 32,
        output_dir: str = "./sft",
        max_steps: int = -1,
        num_train_epochs: int = 3,
        logging_steps: int = 100,
        eval_steps: int = 1000000000000, # skip evaluation
        save_steps: int = 1000000000000, # skip saving
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        learning_rate: float = 5e-5,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.01,
        weight_decay: float = 0.05,
        bf16: bool = True,
        remove_unused_columns: bool = True,
        run_name: str = "sft_webtoon_250409_llama4_109b",
        report_to: str = "wandb",
        gcs_sft_output_dir: str = "gs://us-central1-kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250409",):


    import os
    import subprocess
    import torch
    from accelerate import Accelerator
    from datasets import load_dataset, Dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
    from tqdm import tqdm
    tqdm.pandas()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, \
        TrainingArguments
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
    from google.cloud import bigquery
    import gcsfs
    import shutil
    import wandb
    wandb.login(key='6f5b18e94c20bee59a84b92ca785d1a3acd0f06a')
    print(f'::: torch.cuda.device_count() {torch.cuda.device_count()}:::')

    client = bigquery.Client(project=project_id)
    fs = gcsfs.GCSFileSystem(project=project_id)
    auth_token = "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"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["HUGGING_FACE_HUB_TOKEN"] = auth_token


    if os.path.exists('base-model'):
        shutil.rmtree('base-model')
    os.mkdir('base-model')
    # base model이 GCS에 저장되어 있는 경우
    if base_model_name_or_path.startswith('gs://'):
        subprocess.run(f"gsutil -m cp {base_model_name_or_path}/* ./base-model/", shell=True)
    # base model이 huggingface hub에 있는 경우
    else:
        subprocess.run("huggingface-cli login --token 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", shell=True)
        subprocess.run(
            f'huggingface-cli download {base_model_name_or_path} --local-dir-use-symlinks=False --local-dir=base-model --include "*.safetensors" "*.json" $$',
            shell=True)
    print('** Finsihed Downloading Model Checkpoint ** ')

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj","up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.float32,#torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        'base-model',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        #attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    #moe bug 회피용 (monkey patch)
    orig_scatter_add = torch.Tensor.scatter_add_
    def safe_scatter_add_(self, dim, index, src):
        if self.dtype != src.dtype:
            src = src.to(self.dtype)
        return orig_scatter_add(self, dim, index, src)
    
    torch.Tensor.scatter_add_ = safe_scatter_add_

    
    # prevent ```RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn```
    model.enable_input_require_grads()


    # 기존 tokenizer를 래핑
    class MaxLengthTokenizerWrapper:
        def __init__(self, tokenizer, max_length):
            self._tokenizer = tokenizer
            self.max_length = max_length
    
        def pad(self, *args, **kwargs):
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = self.max_length
            return self._tokenizer.pad(*args, **kwargs)
    
        def __getattr__(self, name):
            return getattr(self._tokenizer, name)
    
        def __call__(self, *args, **kwargs):
            return self._tokenizer(*args, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer = MaxLengthTokenizerWrapper(tokenizer, max_length=8192)
    
    
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
    train_df['text'] = train_df.prompt.progress_apply(lambda x: instruct_structure(x))

    train_dataset = Dataset.from_pandas(train_df[['text']])

    print('::: Dataset Example :::')
    print(train_dataset[0])

    '''
    eval_sql = """          
              select prompt
              from webtoon_translation.sft_dataset
              where data_split in ('valid')
              """
    eval_df = client.query(eval_sql).result().to_dataframe()
    eval_df['text'] = eval_df.prompt.progress_apply(lambda x: instruct_structure(x))
    eval_dataset = Dataset.from_pandas(eval_df[['text']])
    '''

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['text'])):
            output_texts.append(example['text'][i])
        return output_texts

    response_template_with_context = '<|start_header_id|>assistant<|end_header_id|>'
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, 
                                               tokenizer=tokenizer,)

    
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        # eval_steps=eval_steps,
        save_steps=save_steps,
        # evaluation_strategy='steps',
        per_device_train_batch_size=per_device_train_batch_size,
        # per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # eval_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        remove_unused_columns=remove_unused_columns,
        run_name=run_name,
        report_to=report_to,
        ddp_find_unused_parameters=False, # RuntimeError: Expected to mark a variable ready only once.
    )

    #optimizer.train() 호출 무시하기 (Monkey Patch 방식)
    # 기존 optimizer인 AdamW에 train 메서드가 없으면 만들어줌
    if not hasattr(torch.optim.AdamW, "train"):
        def train_stub(self):
            # 그냥 아무 것도 안 함
            pass
    torch.optim.AdamW.train = train_stub
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=False,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # save finetuned model to GCS
    fs.put(f'{output_dir}/*', gcs_sft_output_dir)


if __name__ == '__main__':
    from kfp.kubernetes import add_pod_annotation
    from kfp import dsl

    PIPELINE_NAME = 'webtoon-sft'
    BUCKET_URI = f"gs://kakao-entertainment-cel-applied-ai-prod/bun/"

    PIPELINE_ROOT = f"{BUCKET_URI}/pipeline/{PIPELINE_NAME}"
    PIPELINE_SPEC_PATH = f"yaml/{PIPELINE_NAME}.yaml"

    @dsl.pipeline(name=PIPELINE_NAME,
                  description="webtoon-sft",
                  pipeline_root=PIPELINE_ROOT,
                  )
    def pipeline_func(
        project_id: str = "prod-ai-project",
        # base_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-405B-Instruct",
        # data_sql: str = "where data_split in ('train')",
        base_model_name_or_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        data_sql: str = "where data_split in ('train') and create_date = (select max(create_date) from webtoon_translation.sft_dataset)",
        max_seq_length: int = 8192,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        lora_r: int = 32,
        output_dir: str = "./sft",
        max_steps: int = -1,
        num_train_epochs: int = 3,
        logging_steps: int = 100,
        eval_steps: int = 1000000000000, # skip evaluation
        save_steps: int = 1000000000000, # skip saving
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        learning_rate: float = 5e-5,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.01,
        weight_decay: float = 0.05,
        bf16: bool = True,
        remove_unused_columns: bool = True,
        run_name: str = "sft_webtoon_250409_llama4_109b",
        report_to: str = "wandb",
        gcs_sft_output_dir: str = "gs://us-central1-kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250409",
    ):

        task = train(
            project_id=project_id,
            base_model_name_or_path=base_model_name_or_path,
            data_sql=data_sql,
            max_seq_length=max_seq_length,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_r=lora_r,
            output_dir=output_dir,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            bf16=bf16,
            remove_unused_columns=remove_unused_columns,
            run_name=run_name,
            report_to=report_to,
            gcs_sft_output_dir=gcs_sft_output_dir,
        )

        task.set_accelerator_type("nvidia.com/gpu")
        task.set_accelerator_limit(8)
        task.set_caching_options(False)

        add_pod_annotation(task, "ai-lab-scheduler-name", "bin-packing-scheduler")

    from kfp.v2 import compiler  # noqa: F811
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PIPELINE_SPEC_PATH)

    import kfp
    client = kfp.Client(host='https://3313888af2601658-dot-us-central1.pipelines.googleusercontent.com')
    # credentials = kfp.auth.ServiceAccountTokenVolumeCredentials(path=None)
    # client = kfp.Client(host=host_url, credentials=credentials)
    run = client.create_run_from_pipeline_package(
        PIPELINE_SPEC_PATH,
        namespace="kubeflow",
        arguments={
        },
    )