import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoConfig
from accelerate import Accelerator, infer_auto_device_map

LLAMA_PATH = 'model/llama_405b_quantized'

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

# 모델의 설정을 먼저 불러오기
config = AutoConfig.from_pretrained(LLAMA_PATH)

from transformers.modeling_utils import init_empty_weights  # ✅ 올바른 import

# 아직 모델을 선언하지 않은 상태에서 `config`를 기반으로 빈 모델 생성
# 🚀 가중치를 전혀 로드하지 않는 완전 빈(empty) 모델 생성
with init_empty_weights():
    empty_model = LlamaForCausalLM(config)
print("빈 모델 선언 완료")


num_gpus = torch.cuda.device_count()

# **🚀 GPU 8대에 균등하게 분배하는 수동 device_map 생성**
device_map = {}

# 임베딩 & 출력 레이어는 GPU 0에 배치
device_map["model.embed_tokens"] = 0
device_map["lm_head"] = 1

# 126개 LlamaDecoderLayer를 8개의 GPU에 균등 분배
for i, layer in enumerate(empty_model.model.layers):
    assigned_gpu = i % num_gpus  # GPU 인덱스 0부터 7까지 순차적으로 할당
    device_map[f"model.layers.{i}"] = assigned_gpu

# 마지막 RMSNorm도 마지막 GPU로 배치
device_map["model.norm"] = 2
device_map["model.rotary_emb"] = 3  # Rotary Embedding도 마지막 GPU로

# 모델 로드 (`device_map` 적용)
model = LlamaForCausalLM.from_pretrained(
    LLAMA_PATH,
    quantization_config=bnb_config,
    device_map=device_map,  # 자동 분배된 device_map 적용
    #attn_implementation="flash_attention_2"
)

from transformers import AutoTokenizer

TOKENIZER_PATH = 'model/llama_405b_quantized'

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
# A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
tokenizer.padding_side = "left"

#tokenizer.save_pretrained('model/llama_405b_quantized')

import re
def instruct_structure(prompt,system_prompt=
                       """You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
    input_text, output_text = prompt.split('### target')
    input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
    input_text = re.sub(r"\[[^\]]+\] ", "none ", input_text)
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {input_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


project_id = "prod-ai-project"

from google.cloud import bigquery
client = bigquery.Client(project=project_id)
sql = """select series_id, episode_id, org_input_text, org_output_text, prompt 
        from webtoon_translation.structured_240820_ep_line
        where data_split = 'romance_valid'"""
df = client.query(sql).result().to_dataframe()
from tqdm import tqdm
tqdm.pandas()
df['prompt'] = df['prompt'].progress_apply(lambda x: instruct_structure(x))

sample = df['prompt'][0]

# 입력 변환 및 토큰화
inputs = tokenizer(sample, return_tensors="pt").to("cuda")
print(inputs)


# 모델 추론
with torch.no_grad():
    output = model.generate(**inputs, 
                            max_length=1000,
                            repetition_penalty=1.2
                           )

# 결과 출력
response = tokenizer.decode(output[0], skip_special_tokens=False)
print(response)
