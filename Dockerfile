# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM ubuntu:18.04

# Expose the port the app runs on
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV GCP_PROJECT="genai-413518"
ENV GCLOUD_PROJECT="genai-413518"
ENV GOOGLE_APPLICATION_CREDENTIALS="application_default_credentials.json"
ENV LANGSMITH_API_KEY="ls__b6dedf8d49ae42f0a5d96bd8b4149dd8"
ENV OPENAI_API_KEY="sk-proj-pXEndLTsst7LwTEkJRP7T3BlbkFJg66ZZ8tzdGk83fv8smuP"

# Install system packages
RUN apt-get update && apt-get install -yq \
    curl \
    wget \
    jq \
    vim \
    build-essential \
    libmysqlclient-dev

# Conda dependencies
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9

# Install Miniconda
RUN curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh" && \
    bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b && \
    rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh

# Set path to conda
ENV PATH=/miniconda/bin:${PATH}

# Update conda and install conda packages
RUN conda update -y conda && \
    conda install -c anaconda -y python=${PY_VER}

# Install mysqlclient
RUN conda install -c anaconda -y mysqlclient

# Copy application code
WORKDIR /app
COPY . .

# Install pip dependencies
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
ENTRYPOINT ["streamlit", "run", "Allec_Marketplace_Chat.py", "--server.port=8080", "--server.address=0.0.0.0"]