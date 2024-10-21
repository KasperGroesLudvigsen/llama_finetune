FROM python:3

WORKDIR /usr/src/app

RUN pip install unsloth
# Also get the latest nightly Unsloth!
RUN pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"


COPY . .