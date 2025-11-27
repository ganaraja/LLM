## DSPy POC and MiproV2 Notebooks

This repository includes a basic DSPy notebook and two MIPROv2 training notebooks (one using GPT models, one using local Ollama).

The basic notebook refers to MCP integration which is present in the code at ```src/svlearn_dspy/dspy_mcp.py```

## Notebooks

### 1. dspy_poc.ipynb
Basic DSPy concepts and usage.

### 2. dspy_mipro_trng.ipynb
MiproV2 training implementation using GPT models.

### 3. dspy_mipro_trng_local.ipynb
MiproV2 training implementation using local Ollama models.

## MCP Code

First run the mcp-server using:

```uv run src/svlearn_dspy/mcp_server.py```

Then run the dspy code which access the mcp-server above:

```uv run src/svlearn_dspy/dspy_mcp.py```

## Setup and Running

### Prerequisites
1. Install dependencies:
   ```bash
   uv sync
   ```

2. Select the appropriate kernel in your Jupyter environment

3. Run the notebook cells

### For Local Training (dspy_mipro_trng_local.ipynb)
Before running the local training notebook, ensure you have:

1. Pull the required model:
   ```bash
   ollama pull qwen3:8b
   ```

2. Start the Ollama server:
   ```bash
   ollama serve
   ```

## Conceptual Map of the MiproV2 Optimization Code

1. **Tooling**: You define `google_search` so the agent can fetch evidence.
    
2. **Policy**: A ReAct module learns to interleave reasoning with tool calls.
    
3. **Data**: HotPotQA provides supervised examples (`question`, reference `answer`).
    
4. **Optimization**: MIPROv2 tunes the ReAct setup to improve accuracy on your metric.
    
5. **Persistence**: You save the optimized configuration and later reload it.
    
6. **Evaluation**: You compare original vs optimized programs with a standardized metric.
    
7. **Debugging**: `inspect_history()` lets you read the model's thought/tool sequence.
    

That's the script in a nutshell: define a tool → wire it into ReAct → optimize with MIPROv2 → save/load → evaluate → inspect traces.