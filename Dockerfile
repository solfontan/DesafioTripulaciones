FROM python:3.10-slim-bullseye

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p app

COPY ./app app

WORKDIR app

EXPOSE 5000

CMD [ "python3", "app.py"]