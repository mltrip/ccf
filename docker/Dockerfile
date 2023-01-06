# FROM ubuntu:20.04
FROM python:3.9

# RUN apt-get update && apt-get install -y \
#     # git \
#     python3-pip \
#     python-is-python3 \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/ src/
RUN pip install --no-cache-dir -r src/ccf/requirements.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_data.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_features.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_ml.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_predictions.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_metrics.txt
# RUN pip install --no-cache-dir -r src/ccf/requirements_ui.txt

COPY work/conf work/conf