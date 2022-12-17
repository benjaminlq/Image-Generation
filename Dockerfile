FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone git@github.com:benjaminlq/Image-Generation.git .

RUN pip install --upgrade pip \
    pip install -r requirements.txt \
    pip install -e .

ENTRYPOINT ["streamlit", "run", "./src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]