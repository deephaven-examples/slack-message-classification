FROM ghcr.io/deephaven/server:0.15.1
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY app.d /app.d
