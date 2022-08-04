# Notes and Resources

## Technical resources

Dalle mini techinal explanation: [link](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained--Vmlldzo4NjIxODA)

BART model on hugging face: [link](https://huggingface.co/docs/transformers/model_doc/bart)

VQGAN on hugging face: [link](https://huggingface.co/boris/vqgan_f16_16384)

Notes on transformer in pytorch - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

## The venv:

to use: `source venv/bin/activate`

Setting up a new venv: `python3 -m venv venv`

Where venv is the name of the virtual environment.

Install requirements from pip using:
`python3 -m pip install -r requirements.txt`

## Lambda labs:

For whatever reason, the venv didn't work on lambda labs. This appears intentional.

Lambda cloud instances are available here: https://lambdalabs.com/cloud/dashboard/instances
The account is under gkamer@college.harvard.edu

You can ssh into the instance using:

--> ssh -i First.pem user@host

where First.pem is the permissions file downloaded from lambda and user@host looks like ubuntu@123.456.789

If the permissions file is unprotected, you can run:

--> sudo chmod 600 /path/to/my/key.pem

## Plans

Might be easiest to create the pytorch model by hand using the config description
Then trace hugging face code to load the pytorch model directly

Pytorch transformer - [here](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
--> That's an encoder. Load the encoder config separately, have the model, etc.
Pass outputs to the decoder

Transformer source - https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer

## Technical notes

This insane bug fix for an error involving "recompute scale factor"
[here](https://github.com/openai/DALL-E/issues/54#issuecomment-1092826376)
--> hopefully will be fixed in new pytorch updates

Setting up filesystem on vm:

In order to download DALLE encoder and decoder onto VM, run:
`curl -O https://cdn.openai.com/dall-e/encoder.pkl`
And ditto for the encoder

At present it's also necessary to move the files into the DALL-E folder.





