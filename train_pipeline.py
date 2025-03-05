import os
from typing import NamedTuple
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Input, Output, Dataset, Model
from google.cloud import aiplatform
from datetime import datetime



@dsl.component(
    base_image="asia-northeast3-docker.pkg.dev/dev-ai-project-357507/bun-docker-repo/bun-docker:latest", 
    packages_to_install=["gcsfs"],
    caching=True)
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

@dsl.component(base_image="asia-northeast3-docker.pkg.dev/dev-ai-project-357507/bun-docker-repo/bun-gpu28:latest", packages_to_install=["deepspeed"])
def train_model(input_dataset: Input[Dataset]):
    import os
    import torch
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU 설정

    import subprocess
    gpu_nums = torch.cuda.device_count()
    subprocess.run([
        "accelerate", "launch",
        "--multi_gpu", "--mixed_precision", "fp16", "--num_processes", str(gpu_nums),
        "/app/train_sub.py",  # 런처 스크립트
        "--input_dataset", input_dataset.path,
        "--output_path", ".bun-bucket-test1/outputs/llama3-8b/v12.2.1",
    ], check=True, env=env)

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
