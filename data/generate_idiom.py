import pandas as pd
import openai
from tqdm import tqdm
import random
import argparse

openai.api_key = "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"

def make_response():
    response = openai.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role":"system", "content":"한국어의 관용어는 글자 그대로의 의미가 아니라 내부에 의미가 숨겨져 있는 표현이야."},
            {"role":"user", "content":'이러한 한국어의 관용표현과 관용표현의 의미를 영어로 짧게 번역한 표현을 쌍(구분자는 > 를 활용해줘)으로 100개 생성해줘'}
        ])
    result = response.choices[0].message.content
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    args = parser.parse_args()
    
    for i in tqdm(range(args.num,args.num+100)):
        result = make_response()
        with open(f'/home/jupyter/poetry-env/data/chatgpt_idiom_result/output_{str(i)}.txt', 'w', encoding='utf-8') as file:
            file.writelines(result)

        