FROM python:3.11.3-slim

WORKDIR /code

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python" ]