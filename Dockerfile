# Python + Java in one image. Spark runs on the JVM, so it needs Java.
# Using Docker means nobody has to install Java or Hadoop tools on their laptop,
# and the pipeline runs the same on Windows, Mac and Linux (and in CI later).
FROM python:3.11-slim-bookworm

# git: MLflow records which commit trained each model. The project folder is
# mounted from the host and owned by a different user, so we tell git to trust it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

WORKDIR /app

# Install dependencies first. Docker caches this layer, so changing your code
# doesn't reinstall every package on each build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "pipeline.run_pipeline"]
