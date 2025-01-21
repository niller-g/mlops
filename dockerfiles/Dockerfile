# Use miniconda base image
FROM continuumio/miniconda3

# Create working directory
WORKDIR /app

# Copy environment files
COPY environment.yaml /app

# Create environment
RUN conda env create -f environment.yaml
# Activate by default
RUN echo "conda activate my_mlops_env" >> ~/.bashrc

# Copy the rest of the project
COPY . /app

# Set shell so conda is recognized
SHELL ["conda", "run", "-n", "my_mlops_env", "/bin/bash", "-c"]

# Install pre-commit hooks (optional)
RUN pre-commit install

CMD ["python", "src/mlops/train.py"]
