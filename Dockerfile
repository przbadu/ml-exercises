FROM python:3.12-slim
WORKDIR app
COPY requirements.txt .
RUN pip install -r requirements.txt 
ENV DATA_PATH=/data
ENV MODELS_PATH=/models
COPY scripts .
ENTRYPOINT [python]
