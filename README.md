
# Microsoft Phi 2 for Classification

Refactoring Microsoft Phi 2 LLM for Sequence Classification Task. Training using LoRA and QLoRA approaches using Huggingface trainer

---

**Microsoft's Phi-2 LLM** is a 2.7 Billion parameter model that surpasses even 70B parameter models in some evaluation benchmarks. Considering its relatively small size, it consumes about 5GB of GPU memory for inference and is very fast, especially when run at half precision.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

Above sample code for generation from huggingface. This code is for generating text in an autoregressive fashion. Transformer-Decoder models have shown to be just as good as Transformer-Encoder models for classification tasks (checkout winning solutions in the Kaggle competition: predict the LLM where most winning solutions fine-tuned Llama/Mistral/Zephyr models for classification).

Phi-2 currently doesn't have sequence classification support on HuggingFace AutoModel APIs. I have refactored the above code (for language generation) to perform sequence classification tasks using Huggingface trainer. Further, I implemented LoRA (Low-Rank Adaptation of Large Language Models) to fine-tune this model.

## Sequence Modeling Approach

Summary of steps:

1. Explore above Phi-2's CausalLM model to identify the backbone and head.
2. Find a suitable sequence classification model on HuggingFace source code - PhiForSequenceClassification ([code](https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/phi/modeling_phi.py#L1165)).
3. Modify sequence classification model to set the backbone from Phi-2.
4. Use HuggingFace trainer, implementing LoRA/QLoRA to train and evaluate.

I have attached some code snippets below to explain some of these steps - for the full code, check colab notebooks [Phi-2LoRA](https://colab.research.google.com/drive/1y_CFog1i97Ctwre41kUnKuTGFWgzGWte?usp=sharing) and [Phi2-QLoRA](https://colab.research.google.com/drive/1TTCSSVL2_XRCHnoGBnAApYAXLkck9L-G?usp=sharing).
