# Vision Language Modeling

## Overview
This project implements a vision-language modeling system capable of understanding and answering questions about images. It uses early and mid fusion techniques to integrate visual and textual data, facilitating a comprehensive approach to Visual Question Answering (VQA).

<img width="323" alt="Screenshot 2024-08-08 at 9 43 38 PM" src="https://github.com/user-attachments/assets/32ab49f2-535f-4a5e-a991-6ce010b46709">

## Features
- **Early Fusion Model**: Combines features from image and text before processing through deep learning models.
- **Mid Fusion Model**: Integrates processed features from separate models for image and text.
- **Custom Dataset Handling**: Efficiently manages image and text data using PyTorch's Dataset class.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or later
- PyTorch 1.8 or later
- torchvision
- transformers
- PIL
- pandas

You can install the necessary dependencies with:
```bash
pip install -r requirements.txt
