# Deep Neural Network Optimization Techniques

This repository contains the implementation of various model compression and optimization techniques applied to the RoBERTa base model, using the SquAD 2.0 Dataset. The project explores the effects of quantization, pruning, and knowledge distillation, both individually and in combination with distributed deep learning, to enhance model performance and efficiency.

## Repository Structure

- `quantization.py`: Implements dynamic quantization on the RoBERTa model to reduce its memory usage and enhance inference speed without significantly affecting the accuracy.
- `pruning.py`: Applies pruning techniques to selectively remove weights from the RoBERTa model, aiming to decrease model complexity and increase computational efficiency.
- `knowledge_distillation.py`: Demonstrates the application of knowledge distillation to train a smaller student model that mimics the performance of the larger RoBERTa model while being computationally less demanding.
- `model_compression_ddl.py`: Combines model compression techniques (quantization and pruning) with distributed deep learning to optimize both model size and training time.
- `profiling.py`: Contains code for profiling the different models to analyze performance metrics such as inference time and memory usage.

## Setup and Installation

To run these scripts, ensure you have the following installed:
- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+
- Datasets 1.6+
- TorchMetrics

You can install the necessary libraries using the following command:
```bash
pip install torch transformers datasets torchmetrics
