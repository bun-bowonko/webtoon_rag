import re
from tqdm import tqdm
import csv
import time
from google.cloud import bigquery
from openai import OpenAI

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def instruct_llama_structure(input_text,system_prompt=
                       """You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def instruct_gpt_structure(prompt):
    input_text, output_text = prompt.split('### target')
    input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
    input_text = re.sub(r"\[[^\]]+\] ", "[UNK] ", input_text)
    return input_text

def clean_text(text):
    return re.sub(r'^\d+\s+\[UNK\]\s+', '', text)
def clean_text2(text):
    return re.sub(r'^\d+\s+','', text)

def extract_idiom(text):
    match = re.search(r'<idiom>(.*?)</idiom>', text, re.DOTALL)
    return match.group(1).strip() if match else None

    
if __name__ == "__main__":
    client = bigquery.Client(project="prod-ai-project")
    sql = """select series_id, episode_id, org_input_text, org_output_text, prompt 
            from webtoon_translation.structured_240820_ep_line
            where data_split = 'romance_valid'"""
    df = client.query(sql).result().to_dataframe()
    tqdm.pandas()
    df['gpt_prompt'] = df['prompt'].progress_apply(lambda x: instruct_gpt_structure(x))
    
    GPT_FINE_TUNING_MODEL="ft:gpt-4o-2024-08-06:kakaoent:webtoon-sft-250225:B4j839q0"
    openai_client = OpenAI(
        api_key='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')
    
    GPT_BASE_MODEL = "chatgpt-4o-latest"
    SYSTEM_IDIOM_PROMPT = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "너는 웹툰 한 회차에 대한 한국어 대사 여러 줄을 입력으로 받고, \
                        해당 대사에 포함된 한글 관용어구를 추출하고 이를 영어로 번역하는 일을 수행할 거야. \
                        대신 모든 관용어구를 추출하는 건 아니고 한글을 영어로 번역했을 때 직역이 아닌 의역해야하는 관용어만 추출하고 번역할거야. \
                        구체적으로, 먼저 대사 내의 한국어 관용어구가 있으면 이를 추출하고 영어로 번역해. \
                        그 다음 한국어 관용어와 번역한 영어를 비교해보면서 직역인지 의역인지 판단해. \
                        만약 의역으로 판단되는 경우 의역된 관용표현을 다음과 같이 생성해줘.\
                        '<idiom> 한국어 관용표현 : 의역된 영어 표현 </idiom>'\
                        그리고 한 문장 내에 의역되어야 하는 여러 개의 관용표현이 있다면 아래와 같이 여러 번 생성하면 돼. \
                        '<idiom> 한국어 관용표현1 : 의역된 영어 표현1 </idiom>'\n'<idiom> 한국어 관용표현2 : 의역된 영어 표현2 </idiom>'... \n \
                        그리고 만약 한 문장 내에 의역되어야 하는 관용표현이 없다면 아무것도 생성하지마."
                    }
                ],
    }

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
    
    for data_idx in range(74):
        hints_origin = []
        gpt_prompt = df['gpt_prompt'][data_idx]
        example = gpt_prompt.split('### source')[1].strip()
        example_li = [clean_text(e) for e in example.split('\n')]
        origin_len = len(example_li)

        idiom_completion = openai_client.beta.chat.completions.parse(
            model= GPT_BASE_MODEL,
            messages = [
                SYSTEM_IDIOM_PROMPT,
                {
                    "role":"user",
                    "content" : [{"type" : "text",
                                  "text" : example
                                }],
                }
            ],
            temperature= 0.2,
            top_p = 0.8
        )
        idiom = idiom_completion.choices[0].message.content
        idiom_li = idiom.split('\n')
        for d in idiom_li:
            hints_origin.append(extract_idiom(d))

        glossary = gpt_prompt.split('### source')[0].strip()+'\n'
        if hints_origin:
            for i in range(len(hints_origin)):
                glossary += '• '+hints_origin[i]+'\n'
        input_text = glossary + '\n\n###source\n' + example
        input_prompt = instruct_llama_structure(input_text)
        print(input_prompt)
        
        start_time = time.time()
        # 모델 추론
        with torch.no_grad():
            output = model.generate(input_prompt, sampling_params)
            print(f"{time.time()-start_time:.2f}")
            result = output[0].outputs[0].text.strip()
            result_li = [clean_text2(d) for d in result.split('\n')]
            result_li = result_li[:origin_len]
            #print(result_li)
        
        with open(f"data/llama_result/llama_vllm_idiom_{data_idx}.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['hangle', 'first_translate'])
            for row in  zip(example_li, result_li):
                writer.writerow(row)
