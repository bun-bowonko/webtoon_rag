from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# 모델 다운로드 (weights, config 등 포함)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# 토크나이저 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_name)