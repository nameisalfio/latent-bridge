FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/train.py"]
