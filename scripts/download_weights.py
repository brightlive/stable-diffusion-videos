#!/usr/bin/env python

import os
import shutil
import sys

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import MODEL_CACHE, MODELS, MODEL_VAE

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

vae = AutoencoderKL.from_pretrained(
    MODEL_VAE,
    cache_dir=MODEL_CACHE,
)

for MODEL in MODELS:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL,
        vae=vae,
        cache_dir=MODEL_CACHE,
    )

# pipe = StableDiffusionPipeline.from_pretrained(
#     MODEL_ID,
#     revision="c9211c53404dd6f4cfac5f04f33535892260668e",
#     vae=vae,
#     cache_dir=MODEL_CACHE,
# )
