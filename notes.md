# Notes and Resources

## Technical resources

Dalle mini techinal explanation: [link](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained--Vmlldzo4NjIxODA)

BART model on hugging face: [link](https://huggingface.co/docs/transformers/model_doc/bart)

VQGAN on hugging face: [link](https://huggingface.co/boris/vqgan_f16_16384)

## The venv:

to use: `source venv/bin/activate`

Setting up a new venv: `python3 -m venv venv`

Where venv is the name of the virtual environment.

## Lambda labs:

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






