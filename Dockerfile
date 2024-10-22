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

# pass MODEL_PATH like this in docker build command: docker build --build-arg MY_BUILD_ARG=my_value -t my_image . 
ENV MODEL_PATH=${MODEL_PATH}
ENV HF_TOKEN=${HF_TOKEN}

RUN python3 download_model.py

CMD ["python", "./finetune.py"]
