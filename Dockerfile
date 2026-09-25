# Python + Java in one image. Spark runs on the JVM, so it needs Java.
# Using Docker means nobody has to install Java or Hadoop tools on their laptop,
# and the pipeline runs the same on Windows, Mac and Linux (and in CI later).
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first. Docker caches this layer, so changing your code
# doesn't reinstall every package on each build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "pipeline.run_pipeline"]
