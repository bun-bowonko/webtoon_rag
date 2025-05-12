import kfp
from kfp.v2 import dsl
from typing import NamedTuple


@dsl.component(
    base_image = f"asia-northeast3-docker.pkg.dev/prod-ai-project/tmp-h100/llama3.1-base@sha256:4be245a55e9cb14797d2beb4d58e2f63fbe5c3d186dcfff06a36e6351792cf21",
    packages_to_install=["transformers==4.51.1"],
    output_component_file = 'webtoon-merge-and-save.yaml'
)
def upload(
    project_id: str = "prod-ai-project",
    base_model_name_or_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    finetuned_adapter: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417",
    gcs_output_dir: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417-merged",
    ):

    import subprocess
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import torch
    import os
    import shutil
    from tqdm import tqdm
    tqdm.pandas()
    import logging
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextExperts
    import gc

    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
    logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s',datefmt='%Y-%m-%d:%H:%M:%S',level=logging.WARNING)

    import torch
    import torch.nn as nn
    from transformers.activations import ACT2FN
    
    class Llama4TextExpertsWithLinear(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_local_experts
            self.intermediate_size = config.intermediate_size
            self.hidden_size = config.hidden_size
            self.expert_dim = self.intermediate_size
    
            self.gate_up_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, 2 * self.expert_dim, bias=False)
                for _ in range(self.num_experts)
            ])
            self.down_proj = nn.ModuleList([
                nn.Linear(self.expert_dim, self.hidden_size, bias=False)
                for _ in range(self.num_experts)
            ])
    
            self.act_fn = ACT2FN[config.hidden_act]
        #nn.Linear(in_features, out_features)에서 weight shape은 (out_features, in_features)이므로: T 해야함
        def load_from_param_tensor(self, gate_up_tensor, down_tensor):
            for i in range(self.num_experts):
                # 먼저 Linear 레이어의 weight와 bias를 bfloat16으로 강제 변환
                self.gate_up_proj[i] = self.gate_up_proj[i].to(dtype=torch.bfloat16)
                self.down_proj[i] = self.down_proj[i].to(dtype=torch.bfloat16)

                # 레이어 자체를 복사 대상 텐서의 디바이스로 이동
                gate_up_tensor_device = gate_up_tensor[i].device
                down_tensor_device = down_tensor[i].device
                self.gate_up_proj[i] = self.gate_up_proj[i].to(gate_up_tensor_device)
                self.gate_up_proj[i].weight = self.gate_up_proj[i].weight.to(gate_up_tensor_device)

                self.down_proj[i] = self.down_proj[i].to(down_tensor_device)
                self.down_proj[i].weight = self.down_proj[i].weight.to(down_tensor_device)
                
                # 파라미터 복사 (이미 같은 디바이스에 있음)
                self.gate_up_proj[i].weight.data.copy_(
                    gate_up_tensor[i].T.contiguous()
                )
                self.down_proj[i].weight.data.copy_(
                    down_tensor[i].T.contiguous()
                )

                gate_up_tensor[i] = gate_up_tensor[i].cpu()
                down_tensor[i] = down_tensor[i].cpu()
            # 리스트 항목을 역순으로 삭제하여 인덱스 오류 방지
            for i in range(self.num_experts - 1, -1, -1):
                del gate_up_tensor[i]
                del down_tensor[i]
            del gate_up_tensor
            del down_tensor
            gc.collect()
            torch.cuda.empty_cache()

                        
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            outputs = []
            for i in range(self.num_experts):
                local_input = hidden_states[i].to(self.gate_up_proj[i].weight.device)
     
                gate_up = self.gate_up_proj[i](local_input)
                gate, up = gate_up.chunk(2, dim=-1)
                out = self.down_proj[i](up * self.act_fn(gate))
                outputs.append(out)
            outputs = [out.to("cuda:0") for out in outputs]  # 또는 cat 하기 전에 하나의 device로 모음, 안 그러면 터짐
            next_states = torch.cat(outputs, dim=0)
            return next_states

    
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

    if os.path.exists('finetuned-adapter'):
        shutil.rmtree('finetuned-adapter')
    os.mkdir('finetuned-adapter')
    subprocess.run(f"gsutil -m cp {finetuned_adapter}/* ./finetuned-adapter/", shell=True)
    print('** Finsihed Downloading Adapter Checkpoint ** ')

    model = AutoModelForCausalLM.from_pretrained(
        'base-model',
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    for layer in model.model.layers:
        if not isinstance(layer.feed_forward, Llama4TextMoe):
            continue  # MoE가 아닌 경우 skip
    
        old_expert = layer.feed_forward.experts
    
        # 새로운 Linear 기반 Expert 모듈 생성
        new_expert = Llama4TextExpertsWithLinear(model.config)
    
        # nn.Linear의 weight만 추출해서 넘겨줌
        gate_up_weights = [proj.clone().detach().to(proj.device) for proj in old_expert.gate_up_proj]
        down_weights = [proj.clone().detach().to(proj.device) for proj in old_expert.down_proj]

    
        # weight 복사 & free
        new_expert.load_from_param_tensor(
            gate_up_tensor=gate_up_weights,
            down_tensor=down_weights,
        )
        
        # Expert 모듈 교체
        layer.feed_forward.experts = new_expert
        
        print("파라미터 ✅체크✅")
        for name, param in layer.named_parameters():
            print(f"{name}: {param.shape}, device={param.device}, dtype={param.dtype}")
            
    print("✅✅✅ Llama4TextExperts 계층이 Llama4TextExpertsWithLinear으로 성공적으로 교체되었습니다!")

    
    model = PeftModel.from_pretrained(
        model,
        "finetuned-adapter",
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    print("✅✅✅ merge 완료 & copy 시작")

    def copy_linear_weights_to_param_expert(param_expert, gate_up_linear_layers, down_linear_layers):
        # param_expert 내부 파라미터들 모두 bfloat16으로 변경
        param_expert = param_expert.to(dtype=torch.bfloat16)
        num_experts = len(gate_up_linear_layers)
        
        for i in range(num_experts):
            # weight 복사 (transpose 필요)
            gate_up_weight = gate_up_linear_layers[i].weight.detach().to(torch.bfloat16).T.contiguous()
            down_weight = down_linear_layers[i].weight.detach().to(torch.bfloat16).T.contiguous()
    
            # 디바이스 일치
            device = param_expert.gate_up_proj[i].device
            gate_up_weight = gate_up_weight.to(device)
            down_weight = down_weight.to(device)
    
            # 실제 복사
            param_expert.gate_up_proj[i].data.copy_(gate_up_weight)
            param_expert.down_proj[i].data.copy_(down_weight)
    
    
        # 리스트 메모리 정리
        for i in range(num_experts - 1, -1, -1):
            del gate_up_linear_layers[i]
            del down_linear_layers[i]
        del gate_up_linear_layers
        del down_linear_layers
        gc.collect()
        torch.cuda.empty_cache()

    for layer in model.model.layers:
        if not isinstance(layer.feed_forward.experts, Llama4TextExpertsWithLinear):
            continue
    
        old_expert = layer.feed_forward.experts  # Linear 구조
        new_expert = Llama4TextExperts(model.config) # nn.Parameter 구조
    
        gate_up_linear_layers = [proj for proj in old_expert.gate_up_proj]
        down_linear_layers = [proj for proj in old_expert.down_proj]
    
        copy_linear_weights_to_param_expert(
            param_expert=new_expert,
            gate_up_linear_layers=gate_up_linear_layers,
            down_linear_layers=down_linear_layers,
        )
    
        layer.feed_forward.experts = new_expert
        
        print("파라미터 ✅체크✅")
        for name, param in layer.named_parameters():
            print(f"{name}: {param.shape}, device={param.device}, dtype={param.dtype}")
            
    print("✅✅✅ Llama4TextExpertsWithLinear 계층이 Llama4TextExperts으로 성공적으로 교체되었습니다!")

    
    model.save_pretrained('merged-model', safe_serialization=True)
    print('** Finsihed Saving Model** ')

    # save merged model to GCS
    subprocess.run(f"gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp merged-model/* {gcs_output_dir}/", shell=True)
    print('** Finsihed Uploading Merged Model ** ')


if __name__ == '__main__':

    PIPELINE_NAME = 'webtoon-merge-and-save'
    BUCKET_URI = f"gs://kakao-entertainment-cel-applied-ai-prod/bun/"

    PIPELINE_ROOT = f"{BUCKET_URI}/pipeline/{PIPELINE_NAME}"
    PIPELINE_SPEC_PATH = f"yaml/{PIPELINE_NAME}.yaml"

    @dsl.pipeline(name=PIPELINE_NAME,
                  description="webtoon-merge-and-save",
                  pipeline_root=PIPELINE_ROOT,
                  )
    def pipeline_func(
    project_id: str = "prod-ai-project",
    base_model_name_or_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    finetuned_adapter: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417",
    gcs_output_dir: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417-merged",
    ):

        upload_task = upload(project_id=project_id,
                      base_model_name_or_path=base_model_name_or_path,
                      finetuned_adapter=finetuned_adapter,
                      gcs_output_dir=gcs_output_dir,)

        upload_task.set_accelerator_type("nvidia.com/gpu")
        upload_task.set_accelerator_limit(4)
        upload_task.set_caching_options(False)


    from kfp.v2 import compiler  # noqa: F811
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PIPELINE_SPEC_PATH)

    import kfp
    client = kfp.Client(host='https://3313888af2601658-dot-us-central1.pipelines.googleusercontent.com')
    run = client.create_run_from_pipeline_package(
        PIPELINE_SPEC_PATH,
        arguments={
        },
    )