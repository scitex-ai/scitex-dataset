# hash:sha256:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade --no-cache-dir --upgrade-strategy=only-if-needed \
      networkx==2.1 \
      pandas==0.23.1 \
      scikit-learn==0.19.2
