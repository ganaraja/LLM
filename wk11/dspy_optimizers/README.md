## Overview

This repository contains a quick-start reference plus three task notebooks demonstrating the use of GEPA (Genetic Evolutionary Prompt Architecture) with DSPy for optimizing AI systems:

### Notebook 0: DSPy Optimizers Intro (`optimizers_intro.ipynb`)

Markdown-style lecture notes that walk through DSPy optimizer concepts (few-shot generation, instruction rewriting, finetuning) and catalog the main optimizers such as BootstrapFewShot, GEPA, and MIPROv2, including when to use each and the key parameters to tune.

### Notebook 1: Information Extraction (`gepa_1_info_extraction.ipynb`)

Optimizes a facility support message analyzer using GEPA to extract urgency, sentiment, and categories from facility management messages. Uses the Meta llama-prompt-ops dataset and achieves significant accuracy improvements through automatic prompt optimization with multi-module DSPy programs and custom metrics.

- **MIPROv2 Version**: `miprov2_1_info_extraction.ipynb` demonstrates the same information extraction task using MIPROv2 optimization instead of GEPA.

### Notebook 2: Privacy-Preserving System (`gepa_2_papillon.ipynb`)

Optimizes the PAPILLON framework using GEPA to enable privacy-preserving interactions with external LLMs. The system transforms private queries into anonymized requests, then uses the LLM response to answer the original query. Uses the PUPA dataset from Columbia NLP and employs dual-metric evaluation (quality + PII leakage) to balance response quality with privacy protection.

- **MIPROv2 Version**: `miprov2_2_papillon.ipynb` demonstrates the same privacy-preserving system using MIPROv2 optimization instead of GEPA.

### Notebook 3: Reddit TL;DR Generation (`gepa_3_tldr.ipynb`)

Optimizes a single-module TL;DR generator trained on the `mlabonne/smoltldr` dataset. GEPA tunes the prompt to hit exact 25-word summaries, keep single-line formatting, and maximize semantic fidelity measured via a MiniLM encoder–based metric.

- **MIPROv2 Version**: `miprov2_3_tldr.ipynb` performs the same summarization task using MIPROv2 optimization for comparison.

Both notebooks showcase GEPA's ability to automatically improve prompt engineering through evolutionary optimization, requiring minimal manual intervention while achieving significant performance gains.

## Performance Comparison

The following table compares the performance metrics for both optimization methods (GEPA and MIPROv2) across both tasks:

| Task | Optimizer | Baseline | Optimized | Improvement |
|------|-----------|----------|-----------|-------------|
| Information Extraction | GEPA | 75.78% | 85.44% | +9.66 pp |
| Information Extraction | MIPROv2 | 75.78% | 82.55% | +6.77 pp |
| Privacy-Preserving System (PAPILLON) | GEPA | 80.76% | 84.25% | +3.49 pp |
| Privacy-Preserving System (PAPILLON) | MIPROv2 | 78.96% | 82.17% | +3.21 pp |
| Reddit TL;DR Generation | GEPA | 90.92% | 93.21% | +2.29 pp |
| Reddit TL;DR Generation | MIPROv2 | 90.92% | 91.13% | +0.21 pp |

*pp = percentage points*

**Note on Training Budget of PAPILLON:** The GEPA optimization in this notebook uses `max_full_evals=1` (a single full evaluation), which is a lightweight training budget for demonstration purposes. In contrast, the MIPROv2 papillon code uses `auto="heavy"`, which is much more intensive and performs significantly more evaluations. Despite using a much smaller training budget, GEPA achieves comparable or better performance improvements, demonstrating its efficiency.

## Setup and Running

### Before running the notebooks, ensure you have:

1. Update ```.env``` with your paths and OpenAI API Key

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Select the appropriate kernel in your Jupyter environment

4. Run the notebook cells