FROM python:3.9

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y

COPY ./requirements-deploy.txt ./setup.py ./setup.cfg ./

RUN pip install --upgrade pip \
    pip install -r requirements-deploy.txt --quiet --no-cache-dir

COPY . .

RUN pip install -e .

ENTRYPOINT ["streamlit", "run", "./src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]