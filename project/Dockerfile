FROM python:3.8-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

RUN mkdir /app/output

COPY [ "predict.py", "./" ]
COPY [ "output/be.pkl", "/app/output/be.pkl"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
