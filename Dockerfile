FROM nvcr.io/nvidia/pytorch:24.01-py3

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace
