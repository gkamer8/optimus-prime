import io
import ntpath
import os, sys
import math
import requests
import PIL
import sys
import pandas as pd

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn
from transformers import BartTokenizer

sys.path.insert(0, 'DALL-E')
from dall_e import map_pixels, unmap_pixels, load_model

from custom_work import TransformerModel

target_image_size = 256

ntoken = 50265  # Text tokens (50265)
nimagetoken = 8192  # DALLE tokens (8192)
d_model = 1024  # Model dimension
nhead = 2  # Attention heads in model (16)
d_hid = 2  # FFN dimension (12)
nlayers = 2  # Transformer layers (12)
dropout = 0.1  # Dropout prob (.1)

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

def preprocess(img):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

if __name__ == '__main__':
    device = torch.device('cpu')

    enc = load_model("DALL-E/encoder.pkl", device)
    dec = load_model("DALL-E/decoder.pkl", device)

    print("Finished loading encoder/decoder")

    url = "http://lh6.ggpht.com/-IvRtNLNcG8o/TpFyrudaT6I/AAAAAAAAM6o/_11MuAAKalQ/IMG_3422.JPG?imgmax=800"    
    caption = "a very typical bus station"
    
    tokenizer = BartTokenizer.from_pretrained("model-downloads/bart-tokenizer")
    token_dict = tokenizer([caption])
    tokens = torch.tensor(token_dict['input_ids']).permute((1, 0))
    src_mask = None

    x = preprocess(download_image(url))
    z_logits = enc(x)
    z = torch.argmax(z_logits, axis=1)
    image_torch = torch.flatten(z, start_dim=1, end_dim=2).permute((1, 0))

    tgt_mask = torch.triu(torch.ones(d_model, d_model) * float('-inf'), diagonal=1)  # Additive causal mask

    print("Evaluating Saved Model")
    model = TransformerModel(ntoken=ntoken, nimagetoken=nimagetoken, d_model=d_model, nhead=nhead, d_hid=d_hid,
                    nlayers=nlayers, dropout=dropout)
    model.load_state_dict(torch.load('model2.pt'))
    model.eval()

    starts = torch.zeros((1, 1)).int()
    y = torch.cat((image_torch, starts), 0)
    y_input = y[:-1,:]

    out = model(tokens, y_input, src_mask, tgt_mask)
    output_tokens = torch.argmax(out, axis=2).permute((1, 0)).reshape((1, int(math.sqrt(d_model)), int(math.sqrt(d_model))))
    output_tokens = F.one_hot(output_tokens, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
    x_stats = dec(output_tokens).float()
    
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    print("Not showing")
    
    criterion = nn.CrossEntropyLoss()
    lr = .1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(20):

        starts = torch.zeros((1, 1)).int()
        y = torch.cat((image_torch, starts), 0)  # Put start of sequence at the beginning
        y_input = y[:-1,:]  # Shift input over right
        y_expected = y[1:,:]  # Original image is what gets compared to loss

        out = model(tokens, y_input, src_mask, tgt_mask)
        # Out shape: (seq length, batch size, vocab size)

        print("Finished model inference")

        permuted_image_torch = y_expected.permute((1, 0))
        permuted_out = out.permute((1, 2, 0))

        loss = criterion(permuted_out, permuted_image_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss:")
        print(loss.detach().item())
    
    torch.save(model.state_dict(), 'model2.pt')
    print("Model saved")

    print("Evaluating Saved Model")
    model = TransformerModel(ntoken=ntoken, nimagetoken=nimagetoken, d_model=d_model, nhead=nhead, d_hid=d_hid,
                    nlayers=nlayers, dropout=dropout)
    model.load_state_dict(torch.load('model2.pt'))
    model.eval()

    starts = torch.zeros((1, 1)).int()
    y = torch.cat((image_torch, starts), 0)
    y_input = y[:-1,:]

    out = model(tokens, y_input, src_mask, tgt_mask)
    print(out)
    output_tokens = torch.argmax(out, axis=2).permute((1, 0)).reshape((1, int(math.sqrt(d_model)), int(math.sqrt(d_model))))
    print(output_tokens)
    output_tokens = F.one_hot(output_tokens, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
    x_stats = dec(output_tokens).float()
    
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    print("Still not showing")
    


