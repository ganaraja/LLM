# Getting started with the labs

You will find the lab notes and other related documentation that will help you with your project.

## Config file paths
Update below entries of the `config.yaml` file of the project's root folder to point to suitable paths in your local:

```yaml
models:
  english-sentence-encoder: BAAI/bge-base-en-v1.5   # model for embeddings generation

data:
  chunks_path: /home/devops/ravi/data_hub/chunks # path where the Subject chunks are saved.
```

## Data required for labs
The required data for these can be downloaded from the `llm bootcamp` data hub site.

1. For text chunks: `chunks.tar.gz`
2. Extract using command: `tar -xvzf chunks.tar.gz`