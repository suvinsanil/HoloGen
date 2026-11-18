FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

COPY backend /workspace/backend
COPY scripts /workspace/scripts
COPY datasets/samples /workspace/datasets/samples
COPY models/README.md /workspace/models/README.md

ENV PYTHONPATH=/workspace/backend
ENV HOLOGEN_DEVICE_PREFERENCE=auto

RUN groupadd --system hologen && useradd --system --gid hologen --create-home hologen \
    && chown -R hologen:hologen /workspace
USER hologen

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
