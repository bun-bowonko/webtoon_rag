import re
from tqdm import tqdm
import csv
import time
from google.cloud import bigquery

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def instruct_structure(prompt,system_prompt=
                       """You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
    input_text, output_text = prompt.split('### target')
    input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
    input_text = re.sub(r"\[[^\]]+\] ", "[UNK] ", input_text)
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def clean_text(text):
    return re.sub(r'^\d+\s+\[UNK\]\s+', '', text)
def clean_text2(text):
    return re.sub(r'^\d+\s+','', text)


if __name__ == "__main__":
    model = LLM(
        'model/dpo-240820-ep-line-merged', 
        quantization="bitsandbytes",  # bitsandbytes로 양자화된 모델 로드
        load_format="bitsandbytes",
        tensor_parallel_size=8,  # GPU 사용
        max_model_len=30000,#입력제한
        gpu_memory_utilization=0.995
        
    )
    
    
    TOKENIZER_PATH = 'model/dpo-240820-ep-line-merged'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    sampling_params = SamplingParams(temperature=0.1,
                                     repetition_penalty=1.2,
                                     top_p=0.9,
                                     top_k=30,
                                     max_tokens=2000,
                                     skip_special_tokens=False,
                                     stop = [tokenizer.eos_token]
                                    )
    
    client = bigquery.Client(project="prod-ai-project")
    sql = """select series_id, episode_id, org_input_text, org_output_text, prompt 
            from webtoon_translation.structured_240820_ep_line
            where data_split = 'romance_valid'"""
    df = client.query(sql).result().to_dataframe()
    tqdm.pandas()
    df['prompt'] = df['prompt'].progress_apply(lambda x: instruct_structure(x))
    
    for data_idx in range(74):
        prompt = df['prompt'][data_idx]
        pre = prompt.split('### source')[1].strip()
        pre_li = [clean_text(d) for d in pre.split('\n')]
        pre_li[-1] = pre_li[-1].split('<|eot_id|>')[0]
        pre_len = len(pre_li)
        
        start_time = time.time()
        # 모델 추론
        with torch.no_grad():
            output = model.generate(prompt, sampling_params)
            print(f"{time.time()-start_time:.2f}")
            result = output[0].outputs[0].text.strip()
            result_li = [clean_text2(d) for d in result.split('\n')]
            result_li = result_li[:pre_len]
            #print(result_li)
        
        with open(f"data/llama_result/llama_vllm_{data_idx}.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['hangle', 'first_translate'])
            for row in  zip(pre_li, result_li):
                writer.writerow(row)
