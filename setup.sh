#!/bin/bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -P ~/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/resnet50-11ad3fa6.pth
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
