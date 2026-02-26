# Dockerfile for CLASSIX (multi-arch support)
FROM python:3.10-slim

LABEL maintainer="your@email.com"
LABEL description="CLASSIX clustering library with multi-arch support"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/nla-group/classix.git . 

RUN pip install --no-cache-dir \
    numpy scipy scikit-learn matplotlib threadpoolctl cython \
    && pip install -e .

CMD ["bash"]

