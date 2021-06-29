FROM python:3.8

WORKDIR /src

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "--workers=1", "--threads=1", "-b 0.0.0.0:8050", "app:server"]
