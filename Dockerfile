FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime
# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN pip install unsloth && \
    pip uninstall unsloth -y && \
    pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip uninstall transformers -y && \
    pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git" && \
    pip install codecarbon


COPY . .
CMD ["python", "./finetune.py"]
