# MlOps Project

## Overall goal of the project

The goal is to set up a summarization pipeline for medical text, with particular focus on MLOps best practices.
Rather than aiming for top-tier model performance, the priority is to build a reproducible workflow that can generate context-specific summaries of patient data.

## What framework are you going to use (LangChain, PEFT, etc.)

LangChain (for retrieval-based summarization) and a PEFT library (for parameter-efficient fine-tuning) will be considered.
Both offer methods to adapt a base language model without requiring extensive computational resources. We may opt for something else if we find a better fit.

### How do you intend to include the framework in your project?

With LangChain, a simple chain would be designed to fetch relevant segments of patient notes and produce short summaries.
With PEFT, a smaller Llama model would be fine-tuned on medical terminology in a parameter-efficient manner.
Regardless of the choice, containerization (e.g., Docker) will be used to ensure consistency and ease of deployment.
What data are you going to run on (initially, may change)
The plan is to use a subset of the MIMIC dataset, containing de-identified ICU patient records. If access to MIMIC is not granted, alternative publicly available datasets will be explored. In the absence of suitable real-world data, synthetic data will be generated to maintain progress on the pipeline.

## What deep learning models do you expect to use

A modest Llama-based model is the main candidate, given its open-source nature. Emphasis will be on integrating MLOps components—such as Docker, CI/CD, and data version control—rather than on achieving state-of-the-art results. A parameter-efficient method or prompting strategy will be employed to adapt the model to medical text without excessive resource demands.
