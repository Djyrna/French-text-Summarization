# French Text Summarization and Small Language Model Fine-tuning

## Overview

This project explores **fine-tuning small and efficient language models** for **text summarization in French**, under strict computational constraints (≤16 GB GPU).  

While large-scale models achieve strong results in summarization, they are often impractical for lightweight environments.  
The main goal was to identify a **resource-efficient model** capable of generating high-quality French summaries and to evaluate fine-tuning performance under limited hardware conditions.

---

## Objectives

1. Select and preprocess an appropriate **French summarization dataset**.  
2. Generate synthetic summaries using a **quantized LLM**.  
3. Fine-tune a **small language model (SLM)** for abstractive summarization.  
4. Evaluate models using **ROUGE** and **BERTScore** metrics.  


## Dataset

### Dataset Used
- **OrangeSum (Filtered New Spaces)** – a French summarization dataset by Eddine et al.  
  - 9,020 samples, average article length ≈ 315–350 words  
  - Two tasks: *title generation* and *abstract summarization*

A subset of 5,409 short articles (< 2,900 characters) was selected for faster fine-tuning.

## Data Processing Pipeline

1. **Cleaning:** removed extra whitespace and non-standard characters.  
2. **Filtering:** selected shorter articles to reduce generation time.  
3. **Prompt Engineering:** used *few-shot examples* and a 50-word output limit to improve generation.  
4. **Post-processing:** removed incomplete or too-short summaries (<10 words).  

---

## Models Tested

### Large Language Models for Summary Generation
| Model | Size | Strengths | Limitations |
|--------|------|------------|--------------|
| Jais-13B | 13B | High-quality, multilingual | Too large for free Colab |
| Qwen2.5-7B-Instruct | 7B | Efficient and strong in French | Needs quantization |
| Mistral-7B | 7B | Great summarization | Heavy for 16GB GPU |
| **Qwen2.5-7B-Instruct (chosen)** | 7B | Best trade-off; 4-bit quantization | Slower generation |

- Average generation time: ~30s per summary after optimization.  
- *vLLM* was tested but incompatible with 4-bit quantization.

---

## Synthetic Summary Evaluation

| Metric | Score |
|---------|--------|
| **ROUGE-1** | 0.3098 |
| **ROUGE-2** | 0.1111 |
| **ROUGE-L** | 0.2029 |
| **BERTScore F1** | 0.7200 |

→ Good lexical overlap and strong **semantic similarity** with reference summaries.

## Fine-tuning Phase

### Candidate Small Models
| Model | Params | Strengths | Fine-tuning Strategy |
|--------|---------|------------|----------------------|
| Qwen2.5-0.5B | 500M | Compact, stable | QLoRA |
| Atlas-Chat-2B | 2B | Multilingual | LoRA/QLoRA |
| mT5-Base | 580M | Strong multilingual | LoRA |
| Mistral-7B | 7B | High quality | QLoRA (too heavy) |

**Chosen model:** `Qwen2.5-0.5B`  
Also tested `Qwen2.5-0.5B-Instruct`.

Optimization techniques:
- **QLoRA** for efficient fine-tuning.  
- **fp16** precision for reduced VRAM.  
- **Gradient accumulation** for effective batch scaling.

---

## Results

| Model | Approach | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT F1 | Notes |
|--------|-----------|----------|----------|----------|----------|-------|
| Qwen2.5-0.5B | Fine-tuned | **0.2623** | **0.0844** | **0.1697** | 0.7000 | Best overall performance |
| Qwen2.5-0.5B-Instruct | Fine-tuned | 0.2178 | 0.0589 | 0.1407 | 0.674 | Slightly lower quality |

✅ Fine-tuning improved both **lexical and semantic similarity**.  
✅ Qwen-General outperformed Qwen-Instruct after fine-tuning.  
✅ Training loss reduced to **1.83** with optimal parameters.  

---

## Conclusion

This project successfully explored **resource-efficient text summarization in French** using small and quantized language models.  
The experiments showed that even under strong GPU limitations, **fine-tuned lightweight models** like Qwen2.5-0.5B can achieve solid summarization quality comparable to much larger models.  

Among the tested models, the **Qwen2.5-0.5B base model** provided the best trade-off between **performance, training time, and hardware feasibility**.  
Although larger datasets and longer training time could improve performance further, this work demonstrates the viability of **low-cost multilingual summarization** using modern quantization and fine-tuning techniques.

## 🧰 Technologies Used
- Python (Transformers, PEFT, Evaluate, Datasets)
- QLoRA (Quantized Low-Rank Adaptation)
- Hugging Face Hub (OrangeSum dataset, model hosting)
