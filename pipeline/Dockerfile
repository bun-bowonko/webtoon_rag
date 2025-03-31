FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13

RUN apt-get update && apt-get install -y git

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir transformers datasets peft accelerate google-cloud-aiplatform bitsandbytes gcsfs

COPY train_sub.py /app/train_sub.py
COPY ds_config.json /app/ds_config.json

WORKDIR /app

