from google.cloud import bigquery
import re
from tqdm import tqdm
import pandas as pd

GPT_BASE_MODEL = "chatgpt-4o-latest"

SYSTEM_REVIEW_PROMPT = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "너는 1차 번역된 웹툰 대화를 검토하고 이를 수정하거나 아예 재생성하는 태스크를 수행할 거야. \
                    먼저 맥락을 이해하기 위해 한 회차의 모든 한국어 대화문이 주어지고, \
                    그 중에서 맨 마지막 두 줄에 [한글 대화]와 이 대화에 대응되는 [1차 영어 번역문]이 주어질 거야. \
                    먼저 [1차 영어 번역문]을 한글로 번역해서 [1차 한글 번역문]을 생성하고, [한글 대화]의 의미가 일치하는지 비교해.\n\
                    만약 의미가 일치한다면 \
                    [한글 대화]에서 glossary에서 존재하는 표현은 없는지 검토하고, 만약 있다면 이를 반영해서 \
                    더 나은 [검토한 영어 번역문]을 생성해줘. 물론 [한글 대화]에 glossary에 존재하는 표현도 없고 기존 영어 번역문이 완벽하다면 그대로 생성해도 돼. \
                    하지만 만약 [1차 한글 번역문]과 [한글 대화]의 의미가 일치하지 않거나 정보의 양이 다르다면 \
                    [1차 영어 번역문]과 [1차 한글 번역문]은 무시하고 [한글 대화]에 대응하는 영어 번역문을 생성해.\
                    예를 들어 [한글 대화]가 '사과'인데 [1차 한글 번역문]이 '붉은 사과'라면 \
                    이는 [한글 대화]에 [1차 한글 번역문]의 일부만 제공된 상태이기 때문에 정보의 양이 다른 것으로 판단해. \
                    그렇기 때문에 [1차 한글 번역문]과 [1차 영어 번역문]은 무시하고 [한글 대화]만을 가지고 다시 번역하면 돼.\n \
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

project_id = "prod-ai-project"
client = bigquery.Client(project=project_id)
sql = """select series_id, episode_id, org_input_text, org_output_text, prompt 
        from webtoon_translation.structured_240820_ep_line
        where data_split = 'romance_valid'"""
df = client.query(sql).result().to_dataframe()

tqdm.pandas()
df['prompt'] = df['prompt'].progress_apply(lambda x: instruct_structure(x))

from openai import OpenAI
openai_client = OpenAI(
    api_key='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')

def extract_translation(text):
    match = re.search(r'<translate>(.*?)</translate>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def clean_text(text):
    return re.sub(r'^\d+\s+\[UNK\]\s+', '', text)

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

for data_idx in range(0,10):
    llama_df = pd.read_csv(f"data/llama_result/llama_vllm_idiom_{data_idx}.csv")
    data = df['prompt'][data_idx].split('### source')[1].strip()
    section = [clean_text(d) for d in data.split('\n')]

    #1차 초벌 번역된 결과물이 번역전 원본의 문장 개수와 같은지 체크
    #같지 않다면 원문의 개수에 맞춰서 원문의 한글 추가 (1차 초벌은 빈 상태로 추가) 
    if len(llama_df) < len(section):
        # print(data_idx, len(llama_df), len(section))
        # print(section)
        add_num = len(section) - len(llama_df)
        for i in range(len(llama_df), len(llama_df)+add_num):
            llama_df.loc[i] = [section[i], '']
        print(llama_df)
    elif len(llama_df) > len(section): # 만약 원데이터보다 더 많은 번역을 했을 경우
        print('error occur')

    section = '\n'.join(section)

    idiom_completion = openai_client.beta.chat.completions.parse(
        model= GPT_BASE_MODEL,
        messages = [
            SYSTEM_IDIOM_PROMPT,
            {
                "role":"user",
                "content" : [{"type" : "text",
                              "text" : section
                            }],
            }
        ],
        temperature= 0.2,
        top_p = 0.8
    )
    hints_origin = []
    idiom = idiom_completion.choices[0].message.content
    idiom_li = idiom.split('\n')
    for d in idiom_li:
        hints_origin.append(extract_idiom(d))

    glossary = df['prompt'][data_idx].split('### source')[0].strip()+'\n'
    if hints_origin:
        for i in range(len(hints_origin)):
            glossary += '• '+hints_origin[i]+'\n'
    input_text = glossary + '\n\n###source\n' + section
    
    #한 회차에 대한 모든 초벌 번역에 대해서 검수
    review_li = []
    for i in range(len(llama_df)):
        hangle = llama_df['hangle'][i]
        first_translate = llama_df['first_translate'][i]
        review_text = f"{input_text}\n\n### review\n[한글 대화] {hangle}\n[1차 영어 번역문] {first_translate}\n\n"
        review_completion = openai_client.beta.chat.completions.parse(
            model= GPT_BASE_MODEL,
            messages = [
                SYSTEM_REVIEW_PROMPT,
                {
                    "role":"user",
                    "content" : [{"type" : "text",
                                  "text" : review_text}],
                }
            ],
            temperature=0.2,
            top_p=0.8
        )
        review = review_completion.choices[0].message.content
        print(review)
        review_li.append(extract_translation(review))
    hangles = llama_df['hangle'].tolist()
    first_translates = llama_df['first_translate'].tolist()
    final_df = pd.DataFrame({'hangle':hangles, 'first_translate':first_translates, 'review':review_li, 'idiom':'\n'.join(hints_origin)})
    final_df.to_csv(f'data/llama_result/llama_idiom_review_{data_idx}.csv', index=False)