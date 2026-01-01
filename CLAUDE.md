# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Medical Visual Question Answering (VQA)** research project for the WOA7015 course. The goal is to compare two approaches for binary (Yes/No) VQA on Chest X-rays from the VQA-RAD dataset:

1. **Model A (CNN Baseline)**: ResNet50 + BiLSTM - A traditional deep learning approach combining a CNN image encoder with an LSTM question encoder
2. **Model B (VLM)**: LLaVA-Med with LoRA fine-tuning - A vision-language model approach using parameter-efficient fine-tuning

## Repository Structure

- `AA.ipynb` - CNN baseline implementation (ResNet50 + BiLSTM)
- `LLaVA_Med_VQA_local_12gb.ipynb` - LLaVA-Med VLM implementation with LoRA
- `VQA_RAD/` - Dataset folder containing:
  - `VQA_RAD Dataset Public.json` - Question-answer annotations
  - `VQA_RAD Image Folder/` - Medical images (synpicXXXXX.jpg)
- `LLaVA/` - Local clone of the LLaVA repository (dependency for Model B)

## Key Implementation Details

### Data Filtering (Both Models)
- Filter for **Chest X-rays only** (`image_organ == "chest"`)
- Filter for **closed-ended questions only** (`answer_type == "closed"`)
- Binary labels: Yes=1, No=0
- Results in 477 samples total

### Data Splitting
Both models use **image-level splitting** to prevent data leakage (all questions for the same image go to the same split):
- Train: 80% (~365 samples)
- Val: 10% (~49 samples)
- Test: 10% (~63 samples)
- **Seed: 777** for reproducibility

### Model A: CNN Baseline (`AA.ipynb`)
- Image encoder: ResNet50 (pretrained, frozen for first 5 epochs, then unfrozen)
- Question encoder: BiLSTM with learned embeddings
- Fusion: Concatenation of image and question features
- Classifier: 2-layer MLP
- Training: AdamW optimizer, ReduceLROnPlateau scheduler, early stopping

### Model B: LLaVA-Med (`LLaVA_Med_VQA_local_12gb.ipynb`)
- Base model: `microsoft/llava-med-v1.5-mistral-7b`
- 4-bit quantization (BitsAndBytes) for ~12GB VRAM
- LoRA fine-tuning (r=16, alpha=32) targeting attention and MLP layers
- Conversation mode: `mistral_instruct`
- System prompt forces binary Yes/No answers

## Running the Notebooks

### Prerequisites
```bash
# For Model A (CNN)
pip install torch torchvision scikit-learn matplotlib

# For Model B (LLaVA-Med)
pip install -e ./LLaVA
pip install bitsandbytes>=0.41.0 peft>=0.7.0 scikit-learn tqdm seaborn
```

### Environment Variables (Model B)
- `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` - For faster model downloads
- `LLAVA_MED_MODEL_PATH` - Optional: path to local model weights

### GPU Requirements
- Model A: Any CUDA GPU with ~4GB VRAM
- Model B: CUDA GPU with ~12GB VRAM (uses 4-bit quantization)

## Current Results

| Model | Test Accuracy | Test Macro-F1 |
|-------|--------------|---------------|
| ResNet50 + BiLSTM | 0.5556 | 0.5288 |
| LLaVA-Med (LoRA) | TBD | TBD |

## Key Functions

### Model A
- `normalize_answer(ans)` - Converts Yes/No to 1/0
- `image_level_split(samples, seed)` - Splits data by image ID
- `ResNetBiLSTMVQA` - Main model class
- `freeze_cnn(model)` / `unfreeze_cnn(model)` - Transfer learning helpers

### Model B
- `create_prompt(question)` - Formats question with system prompt
- `prepare_inputs(image_path, question, ...)` - Prepares model inputs
- `generate_answer(image_path, question, ...)` - Runs inference
- `parse_yes_no(text)` - Extracts binary prediction from model output
