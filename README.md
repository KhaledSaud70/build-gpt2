# GPT-2 from Scratch

A minimal implementation of GPT-2 trained on TinyShakespeare dataset, following Andrej Karpathy's tutorial.

## Installation

```bash
pip install torch numpy requests tiktoken
```

## Usage

Single GPU/CPU:
```bash
python train.py
```

Multi-GPU (DDP):
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

## Project Structure
```
.
├── model.py   # GPT-2 model architecture
├── train.py   # Training script
└── data/      # Dataset directory
```

## Model Configuration

```python
GPTConfig(
    block_size=1024,  # sequence length
    vocab_size=50257, # GPT-2 vocab size
    n_layer=12,       # transformer blocks
    n_head=12,        # attention heads
    n_embd=384       # embedding dimension
)
```

## Features
- Flash Attention support
- Distributed training (DDP)
- Gradient accumulation
- Regular validation and checkpointing

## Acknowledgments
This implementation is based on Andrej Karpathy's tutorial on building GPT from scratch. 
