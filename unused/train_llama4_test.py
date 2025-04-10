import torch
from transformers import AutoModelForCausalLM,Llama4ForConditionalGeneration,LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoConfig
from accelerate import Accelerator, infer_auto_device_map

LLAMA_PATH = "/home/bun.2/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"


#멍키패치라서 추후 설명필요
from transformers.integrations import tensor_parallel

# ✅ bfloat16 dtype이 없으면 수동으로 추가
if torch.bfloat16 not in tensor_parallel.str_to_torch_dtype:
    tensor_parallel.str_to_torch_dtype[torch.bfloat16] = torch.bfloat16

# ✅ get_dtype 에러 방지 패치
if not hasattr(torch.Tensor, "get_dtype"):
    torch.Tensor.get_dtype = lambda self: self.dtype
    
# 4bit 퀀타이제이션 설정
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True  # CPU 오프로딩 활성화
)
# 각 GPU에 최대 메모리 설정 (예시로 40GB씩 할당)
max_memory = {i: "40GB" for i in range(torch.cuda.device_count())}

# 모델 로드 (`device_map` 적용)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    quantization_config=bnb_config,
    device_map="auto",  # 자동 분배된 device_map 적용
    attn_implementation="flash_attention_2",
    max_memory=max_memory
)
