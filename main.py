from transformers import BartTokenizer
import os
from torch import nn
import torch

from transformers.models.bart.modeling_bart import BartEncoder, BartModel
def download_all_used_models(download_folder='model-downloads'):
    download_bart_tokenizer(download_folder)
    
def download_bart_tokenizer(download_folder='model-downloads'):
    # Bart tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_tokenizer_path = os.path.join(download_folder, 'bart-tokenizer')
    tokenizer.save_pretrained(bart_tokenizer_path)

def download_bart(download_folder='model-downloads'):
    # Bart model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    bart_model_path = os.path.join(download_folder, 'bart')
    model.save_pretrained(bart_model_path)


download_all_used_models()
"""tokenizer = BartTokenizer.from_pretrained("model-downloads/bart-tokenizer")
model = SpecialBartModel.from_pretrained("model-downloads/bart")

example = "There is no just cause for an invasion of Iraq."
batch = tokenizer(example, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
"""
