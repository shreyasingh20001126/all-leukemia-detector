FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN python3 -c "from huggingface_hub import hf_hub_download; \
import shutil, os; \
os.makedirs('/app/weights', exist_ok=True); \
files = ['AlexNet.weights.h5','VGG16.weights.h5','ResNet50.weights.h5','ResNet152.weights.h5','MobileNet.weights.h5','EfficientNetV2.weights.h5']; \
[shutil.copy(hf_hub_download(repo_id='shreyasingh2000/all-leukemia-weights', filename=f), f'/app/weights/{f}') for f in files]"

COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
