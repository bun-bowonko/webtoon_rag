import re
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import bs4
import ssl
import urllib3
import pandas as pd
import faiss
import os
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import csv
import gc
from tqdm import tqdm
from google.cloud import bigquery
from openai import OpenAI
import argparse

project_id = "prod-ai-project"

GPT_FINE_TUNING_MODEL="ft:gpt-4o-2024-08-06:kakaoent:webtoon-sft-250225:B4j839q0"
GPT_BASE_MODEL = "chatgpt-4o-latest"
openai_client = OpenAI(
    api_key='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')

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
                    '<idiom> 한국어 관용표현1 : 의역된 영어 표현1 </idiom>\n'<idiom> 한국어 관용표현2 : 의역된 영어 표현2 </idiom>'... \n \
                    그리고 만약 한 문장 내에 의역되어야 하는 관용표현이 없다면 아무것도 생성하지마."
                }
            ],
}

SYSTEM_PROMPT = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
            • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
            • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
            • Translate with an American audience in mind. This means easy-to-read, conversational English.""",
                }
            ],
}

SYSTEM_REVIEW_PROMPT = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "너는 1차 번역된 웹툰 대화를 검토하고 이를 수정하거나 아예 재생성하는 태스크를 수행할 거야. \
                    먼저 맥락을 이해하기 위해 한 회차의 모든 한국어 대화문이 주어지고, \
                    그 중에서 맨 마지막 두 줄에 [한글 대화]와 이 대화에 대응되는 [1차 영어 번역문]이 주어질 거야. \
                    먼저 [1차 영어 번역문]을 한글로 번역한 후 [1차 한글 번역문]을 생성해 그리고 [1차 한글 번역문]과 [한글 대화]의 의미가 일치하는지 비교해.\n\
                    만약 의미가 일치한다면 주어진 [1차 영어 번역문]을 주어진 [한글 대화]와 비교해서 누락되거나 추가된 부분은 없는지, \
                    너무 직역된 부분은 없는지, glossary에서 참조할 만한 건 없는지 검토하고, \
                    더 나은 하나의 영어 번역문을 생성해줘. 물론 기존 영어 번역문이 완벽하다면 그대로 생성해도 돼. \
                    하지만 만약 [1차 한글 번역문]과 [한글 대화]의 의미가 일치하지 않다면 [1차 한글 번역문]은 무시하고 [한글 대화]에 대응하는 영어 번역문을 생성해.\n \
                    그리고 새롭게 생성한 영어 번역문은 <translate>로 시작하고 </translate>로 끝내, \
                    만약 추론이 필요하다면 <reasoning>으로 시작해서 추론 내용을 입력하고 </reasoning>으로 마무리해. \
                    추론할 때는 이 대사의 주어가 무엇인지도 생각해. 필요없으면 주어가 없어도 돼.\
                    ***important*** 그리고 출력에 다음 대사나 이전 대사의 내용을 추가하지말고, 주어진 [한글 대화]의 내용만으로 번역문을 구성해. \
                    쉽게 생각하면 너가 생성한 영어 번역문을 한글로 번역했을 때 주어진 [한글 대화]와 의미가 일치해야해\n \
                    즉 정리하자면 출력 형태는 '<reasoning> ...추론 내용... </reasoning> <translate> 검토한 영어 번역문 </translate>' \
                    혹은 '<translate> 검토한 영어 번역문 </translate>' 가 될 거야"
                }
            ],
}

def extract_idiom(text):
    match = re.search(r'<idiom>(.*?)</idiom>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def instruct_structure(prompt):
    input_text, output_text = prompt.split('### target')
    input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
    input_text = re.sub(r"\[[^\]]+\] ", "[UNK] ", input_text)
    return input_text

def extract_translation(text):
    match = re.search(r'<translate>(.*?)</translate>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def clean_text(text):
    return re.sub(r'^\d+\s+\[UNK\]\s+', '', text)
def clean_text2(text):
    return re.sub(r'^\d+\s+','', text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_chapter_num", default=0, type=int)
    parser.add_argument("--end_chapter_num", default=73, type=int)
    parser.add_argument("--save_path", default='data/testset', type=str)
    args = parser.parse_args()
    
    client = bigquery.Client(project=project_id)
    sql = """select series_id, episode_id, org_input_text, org_output_text, prompt 
        from webtoon_translation.structured_240820_ep_line
        where data_split = 'romance_valid'"""
    df = client.query(sql).result().to_dataframe()
    tqdm.pandas()
    df['prompt'] = df['prompt'].progress_apply(lambda x: instruct_structure(x))
    answers = df['org_output_text']
    
    total_source = []
    total_first = []
    total_second = []

    data_num = len(df['prompt'])
    for data_idx in tqdm(range(args.start_chapter_num, args.end_chapter_num+1)):
        data = df['prompt'][data_idx]
        example = data.split("### source")[1].strip()

        hints_origin = []
        #관용어 추출 inference
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

        prompt = data.split("### source")[0].strip()+'\n'
        for i in range(len(hints_origin)):
            prompt += '• '+hints_origin[i]+'\n'
        input_text = prompt + '\n\n###source\n' + example

        source = input_text.split("###source")[1].strip().split('\n')
        #관용어구 적용 후
        check = False
        while not check:
            #기존 튜닝 모델 inference
            chat_completion = openai_client.beta.chat.completions.parse(
                model= GPT_FINE_TUNING_MODEL,
                messages = [
                    SYSTEM_PROMPT,
                    {
                        "role":"user",
                        "content" : [{"type" : "text",
                                      "text" : input_text
                                    }],
                    }
                ],
                temperature= 0.2,
                top_p = 0.8
            )
            response = chat_completion.choices[0].message.content
            idiom_response_split = response.split('\n')
            check = len(idiom_response_split) == len(source) 
        assert len(idiom_response_split) == len(source) #에러가 날 확률이 있음 > 위 코드 루프로 돌게 만듦

        #관용어구 적용 이전
        check = False
        while not check:
            #기존 튜닝 모델 inference
            chat_completion = openai_client.beta.chat.completions.parse(
                model= GPT_FINE_TUNING_MODEL,
                messages = [
                    SYSTEM_PROMPT,
                    {
                        "role":"user",
                        "content" : [{"type" : "text",
                                      "text" : data
                                    }],
                    }
                ],
                temperature= 0.2,
                top_p = 0.8
            )
            response = chat_completion.choices[0].message.content
            response_split = response.split('\n')
            check = len(response_split) == len(source) 
        assert len(response_split) == len(source) #에러가 날 확률이 있음 > 위 코드 루프로 돌게 만듦
        
        hangle = [clean_text(source[i]) for i in range(len(response_split))]
        first_translate = [clean_text2(d) for d in response_split]
        second_translate = [clean_text2(d) for d in idiom_response_split]
       
        label = answers[data_idx]
        idioms = ['\n'.join(hints_origin)]*len(hangle)
        with open(f"{args.save_path}/idiom_result_{data_idx}.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['hangle', 'first_translate', 'idiom_translate', 'label','idiom'])
            for row in zip(hangle, first_translate, second_translate, label, idioms):
                writer.writerow(row)
                

        


