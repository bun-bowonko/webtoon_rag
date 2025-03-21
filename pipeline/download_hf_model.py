from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./qwen2.5-it-32b", safe_serialization=True)
tokenizer.save_pretrained("./qwen2.5-it-32b")