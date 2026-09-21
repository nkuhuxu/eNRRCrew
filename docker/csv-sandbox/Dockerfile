FROM python:3.12.7-slim

COPY requirements.lock.txt /tmp/requirements.lock.txt

RUN useradd --create-home --uid 10001 analyst \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock.txt

COPY runner.py /sandbox_runner.py

USER analyst
WORKDIR /workspace
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "/sandbox_runner.py"]
