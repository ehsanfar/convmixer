#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
from timm.optim import create_optimizer_v2, optimizer_kwargs

model = create_model("convmixer_1536_20", pretrained=False, num_classes=1000)
# print(model)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.fc = torch.nn.Linear(in_features=1536, out_features=8, bias=True)
checkpoint = torch.load("/home/etanfar/Documents/convmixer/pytorch-image-models/output/train/20230926-223218-convmixer_1536_20-574/model_best.pth.tar")
# print(checkpoint)
model.load_state_dict(checkpoint['state_dict'])

# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
# loss = checkpoint['loss']