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
        run_name: str = "sft_webtoon_250414_llama4_109b",
        report_to: str = "wandb",
        gcs_sft_output_dir: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250414",):


    import os
    import subprocess
    import torch
    from accelerate import Accelerator
    from datasets import load_dataset, Dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
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
    import glob
    wandb.login(key='6f5b18e94c20bee59a84b92ca785d1a3acd0f06a')

    print(f'::: torch.cuda.device_count() {torch.cuda.device_count()}:::')

    client = bigquery.Client(project=project_id)
    fs = gcsfs.GCSFileSystem(project=project_id)
    auth_token = "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"
    #pytorch memory segmentation 방지
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False,max_split_size_mb:1024"
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
        result = subprocess.run(
            f'huggingface-cli download {base_model_name_or_path} --local-dir-use-symlinks=False --local-dir=base-model --include "*.safetensors" "*.json" $$',
            shell=True)
        if result.returncode == 0:
            print("✅ 다운로드 성공!")
        else:
            print(f"❌ 다운로드 실패! returncode: {result.returncode}")
    print('** Finsihed Downloading Model Checkpoint ** ')

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj","up_proj", "down_proj", "gate_proj","router"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,#torch.float32
    )

    # torch.cuda.empty_cache()
    # from accelerate import infer_auto_device_map, init_empty_weights
    # # 1. 빈 모델 먼저 로드
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_pretrained(
    #         'base-model',
    #         attn_implementation="flash_attention_2",
    #         torch_dtype=torch.bfloat16,
    #         device_map=None,
    #         trust_remote_code=True,
    #     )
    # torch.cuda.empty_cache()
    # # 2. device_map 추론
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={
    #         i: "80GiB" for i in range(8)  # H100 80GB 기준
    #     },
    #     no_split_module_classes=["Llama4TextDecoderLayer", "Llama4VisionAttention"],  # 중요!
    # )
    
    # device_map = {}
    # num_layers = 48
    # layers_per_gpu = num_layers // 8
    
    # for i in range(8):
    #     start = i * layers_per_gpu
    #     end = (i + 1) * layers_per_gpu
    #     for j in range(start, end):
    #         device_map[f"model.layers.{j}"] = i
    
    # device_map.update({
    #     "model.embed_tokens": 0,
    #     "model.norm": 7,
    #     "model.rotary_emb": 7,
    #     "lm_head": 7,
    # })
    
    #print(device_map)
    model = AutoModelForCausalLM.from_pretrained(
        'base-model',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)  # LoRA 레이어 생성됨!
    model.print_trainable_parameters()

    def print_gpu_memory():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: Allocated = {allocated:.2f} GB, Reserved = {reserved:.2f} GB")

    # 모델 로딩 직후 호출
    print_gpu_memory()
    
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
    tokenizer.pad_token = tokenizer.eos_token #빼야됨, llama4는 pad토큰 가지고 있음 -> 그래도 상관없음(내부로직에 따라)
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    #tokenizer = MaxLengthTokenizerWrapper(tokenizer, max_length=max_seq_length)
    
    
    def instruct_structure(prompt, system_prompt="""You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
        input_text, output_text = prompt.split('### target')
        input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
        return f"""<|begin_of_text|><|header_start|>system<|header_end|>
        
{system_prompt}<|eot|><|header_start|>user<|header_end|>

{input_text.strip()}<|eot|><|header_start|>assistant<|header_end|>

{output_text.strip()}<|eot|>"""
    
    train_sql = f"""          
              select prompt
              from webtoon_translation.sft_dataset
              {data_sql}
              """
    train_df = client.query(train_sql).result().to_dataframe()
    train_df['text'] = train_df.prompt.progress_apply(lambda x: instruct_structure(x))

    train_dataset = Dataset.from_pandas(train_df[['text']])

    #print('::: Dataset Example :::')
    #print(train_dataset[0])

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

    response_template_with_context = '<|header_start|>assistant<|header_end|>'
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


    class SafeSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            device = model.device
            num_items_in_batch = kwargs.get("num_items_in_batch", None)
    
            # inputs 내 tensor들을 올바른 디바이스로 이동
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
    
            outputs = model(**inputs)
    
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
    
            if num_items_in_batch is not None:
                if isinstance(num_items_in_batch, torch.Tensor):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
    
            return (loss, outputs) if return_outputs else loss
            
    trainer = SafeSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        #peft_config=peft_config,
        tokenizer=tokenizer,
        packing=False,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # save finetuned model to GCS
    # 로컬 디렉토리의 모든 파일 가져오기
    local_files = glob.glob(os.path.join(output_dir, '*'))
    # 파일 하나씩 업로드
    for local_file in local_files:
        filename = os.path.basename(local_file)
        gcs_path = f"{gcs_sft_output_dir}/{filename}"
        fs.put(local_file, gcs_path)
        print(f"Uploaded {local_file} to {gcs_path}")

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
        max_seq_length: int = 4096,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        lora_r: int = 32,
        output_dir: str = "./sft",
        max_steps: int = -1,
        num_train_epochs: int = 3,
        logging_steps: int = 1,
        eval_steps: int = 1000000000000, # skip evaluation
        save_steps: int = 1000000000000, # skip saving
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        gradient_checkpointing: bool = True,
        learning_rate: float = 5e-5,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.01,
        weight_decay: float = 0.05,
        bf16: bool = True,
        remove_unused_columns: bool = True,
        run_name: str = "sft_webtoon_250414_llama4_109b",
        report_to: str = "wandb",
        gcs_sft_output_dir: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250414",
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