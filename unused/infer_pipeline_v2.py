import kfp
from kfp.v2 import dsl
from typing import NamedTuple


@dsl.component(
    base_image = f"asia-northeast3-docker.pkg.dev/prod-ai-project/tmp-h100/llama3.1-base@sha256:8125b2d27a8382d644041b37610ca65f4fa7bf61d5461f5665b34e6f190ab966",
    packages_to_install=["transformers==4.51.1"],
    output_component_file = 'webtoon-evaluation.yaml'
)
def inference(
        project_id: str = "prod-ai-project",
        base_model_name_or_path: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417-merged",
        batch_size: int = 1,
        output_table_name: str = "llama_sft_250417_evaluation_parsed",
    ):

    from transformers import pipeline
    from transformers import BitsAndBytesConfig
    from datasets import load_dataset, Dataset
    from transformers.pipelines.pt_utils import KeyDataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import subprocess
    from google.cloud import bigquery
    import os
    import shutil
    import re
    from tqdm import tqdm
    tqdm.pandas()
    import logging
    client = bigquery.Client(project=project_id)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
    logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.WARNING)

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

    def parse_prediction(prompt, prediction):
        prompt_nums = []
        for line in prompt.split('\n'):
            line = line.strip()
            if not re.findall('^[0-9]{3}', line):
                continue
            prompt_nums.append(int(re.findall('^[0-9]{3}', line)[0]))

        prompt_max_num = max(prompt_nums)

        eng_dict = {i: '' for i in range(prompt_max_num + 1)}

        for line in prediction.split('\n'):
            line = line.strip()
            if not re.findall('^[0-9]{3}', line):
                continue
            key = int(re.findall('^[0-9]{3}', line)[0])
            value = re.sub('^[0-9]{3}', '', line).strip()
            # value = re.sub(r"\[.*?\]", "", re.sub('^[0-9]{3}', '', line).strip()).strip()
            if key in eng_dict:
                eng_dict[key] = value

        return list(eng_dict.values())


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        'base-model',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    tokenizer.padding_side = "left"

    def instruct_structure(prompt, system_prompt="""You're an expert translator who translates Korean webtoon in English. Make sure the number of target sentences matches the number of source sentences. The result should be TSV formatted. 
    • Find a balance between staying true to the Korean meaning and keeping a natural flow. Don't be afraid to add to the text. Embellish it. 
    • Avoid translating word-for-word. Keep the general feeling and translate the text accordingly. 
    • Translate with an American audience in mind. This means easy-to-read, conversational English."""):
        input_text, output_text = prompt.split('### target')
        input_text = input_text.replace('### glossaries', '### glossary').replace('\n* ', '\n• ')
        input_text = re.sub(r"\[[^\]]+\] ", "none ", input_text)
        return f"""<|begin_of_text|><|header_start|>system<|header_end|>
        
{system_prompt}<|eot|><|header_start|>user<|header_end|>

{input_text.strip()}<|eot|><|header_start|>assistant<|header_end|>"""

    sql = """
          select series_id, episode_id, org_input_text, org_output_text, prompt 
          from webtoon_translation.structured_240820_ep_line
          where data_split = 'romance_valid'
          """
    df = client.query(sql).result().to_dataframe()
    df['prompt'] = df.prompt.progress_apply(lambda x: instruct_structure(x))
    infer_dataset = Dataset.from_pandas(df[['prompt']], split="test")

    # p00 parameter와 동일함
    pipe00 = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        top_k=30,
        repetition_penalty=1.2,
    )

    predictions = []
    cnt = 0
    for result in tqdm(pipe00(KeyDataset(infer_dataset, "prompt"), batch_size=batch_size, return_full_text=False)):
        logging.info(cnt)
        pred = result[0]['generated_text'].lower()
        predictions.append(pred)
        if cnt == 0:
            logging.info(pred)
        cnt += 1

    df['prediction'] = predictions
    df['source'] = 'llama_p00'
    df['model_path'] = base_model_name_or_path

    df['parsed_prediction'] = df.progress_apply(lambda x: parse_prediction(x['prompt'].split('### target')[0], x['prediction']), axis=1)

    table_id = f'{project_id}.webtoon_translation.{output_table_name}'
    job_config = bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')
    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    job.result()


if __name__ == '__main__':
    from kfp.kubernetes import add_pod_annotation

    PIPELINE_NAME = 'webtoon-evaluation'
    BUCKET_URI = f"gs://kakao-entertainment-cel-applied-ai-prod/bun/"

    PIPELINE_ROOT = f"{BUCKET_URI}/pipeline/{PIPELINE_NAME}"
    PIPELINE_SPEC_PATH = f"yaml/{PIPELINE_NAME}.yaml"

    @dsl.pipeline(name=PIPELINE_NAME,
                  description="webtoon-evaluation",
                  pipeline_root=PIPELINE_ROOT,
                  )
    def pipeline_func(
            project_id: str = "prod-ai-project",
            base_model_name_or_path: str = "gs://kakao-entertainment-cel-applied-ai-prod/bun/llama4/llama4_109b/sft-webtoon-250417-merged",
            batch_size: int = 1,
            output_table_name: str = "llama_sft_250417_evaluation_parsed",
    ):

        task = inference(project_id=project_id,
                          base_model_name_or_path=base_model_name_or_path,
                          batch_size=batch_size,
                          output_table_name=output_table_name,)

        task.set_accelerator_type("nvidia.com/gpu")
        task.set_accelerator_limit(4)
        task.set_caching_options(False)

        add_pod_annotation(task, "ai-lab-scheduler-name", "bin-packing-scheduler")


    from kfp.v2 import compiler  # noqa: F811
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PIPELINE_SPEC_PATH)

    import kfp
    client = kfp.Client(host='https://3313888af2601658-dot-us-central1.pipelines.googleusercontent.com')
    run = client.create_run_from_pipeline_package(
        PIPELINE_SPEC_PATH,
        arguments={
        },
    )