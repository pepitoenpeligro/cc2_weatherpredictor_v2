# docker build --no-cache -t pepitoenpeligro/cc2_weatherpredictor_v2 -f Dockerfile .
# docker run -p 3006:3006 pepitoenpeligro/cc2_weatherpredictor_v2
# docker run -it -p 3006:3006 pepitoenpeligro/cc2_weatherpredictor_v2 /bin/bash
FROM python:3.8.5-slim

LABEL pepitoenpeligro.cc2_weatherprediction_v2.version="0.1.0"
LABEL pepitoenpeligro.cc2_weatherprediction_v2.release-date="2021-05-06"
LABEL pepitoenpeligro.cc2_weatherprediction_v2.url="https://github.com/pepitoenpeligro/cc2_weatherpredictor_v2"

WORKDIR /app

COPY . ./

EXPOSE 3006

RUN apt-get update && \
    pip install --upgrade pip && \
    pip install --requirement requirements.txt

CMD gunicorn --bind 0.0.0.0:3006 server:app --timeout 60000 --workers=1 --capture-output