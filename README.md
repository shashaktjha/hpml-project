# HPML Project: Model Compression Techniques on Deep Neural Networks

## Project Description
This project focuses on evaluating the impact of model compression techniques on the performance of Transformer-based large language models (LLMs), specifically using RoBERTa. The primary goal is to investigate how techniques like quantization, pruning, and knowledge distillation can reduce the size and computational demand of these models without significantly compromising their performance.

## Project Milestones

### Milestone 1: Setup and Baseline Measurement
- **Status:** Completed
- **Activities:** Configured the RoBERTa model and measured baseline performance metrics.

### Milestone 2: Implement Quantization
- **Status:** Completed
- **Activities:** Applied post-training dynamic quantization and measured the impact on inference time and model size.

### Milestone 3: Implement Pruning
- **Status:** Completed
- **Activities:** Applied unstructured pruning at various levels (30%, 60%, 90%) to the RoBERTa model and analyzed performance.

### Milestone 4: Implement Knowledge Distillation
- **Status:** Completed
- **Activities:** Developed a knowledge distillation setup from a larger teacher model to a smaller student model.

### Milestone 5: Evaluation and Analysis
- **Status:** Completed
- **Activities:** Evaluated and compared the effects of each compression technique on inference time and output accuracy.

## Repository and Code Structure
- `Knowledge_Distillation.py`: Script for setting up and evaluating the teacher-student knowledge distillation.
- `model_compression_ddl.py`: Contains functions for data preparation, model training with pruning, and evaluation.
- `profiling.py`: Includes performance profiling for quantized models to understand runtime behavior and resource utilization.
- `pruning.py`: Script for applying and evaluating different levels of pruning.
- `quantization.py`: Implements and evaluates the model quantization process.
- `bash.sh`: Bash script for running the project on the NYU HPC cluster with SLURM job scheduling.

## Example Commands to Execute the Code
```bash
sbatch bash.sh
'''

Results
Inference Time Improvements
Baseline Inference Time: GPU: 0.142559s, CPU: 0.227092s
Quantization: Reduced inference time to GPU: 0.113439s, CPU: 0.163114s
Pruning: Varied inference times based on the degree of pruning, with slight improvements over baseline.
Knowledge Distillation: Increased inference times due to training overhead but reduced when compared to the teacher model alone.

Observations:
Quantization proved effective in reducing both the model size and inference times across both CPU and GPU, showcasing its utility for deployment in resource-constrained environments.
Pruning showed less consistent results with slight fluctuations in inference times, suggesting a dependency on the degree of pruning and the nature of the data.
Knowledge Distillation resulted in a more compact model with reduced inference capabilities compared to the original full-sized model but offered substantial efficiency improvements over the non-distilled version.



![image](https://github.com/shashaktjha/hpml-project/assets/56186071/04a95518-077c-447c-a8b1-a2aafc6635dc)

![image](https://github.com/shashaktjha/hpml-project/assets/56186071/fad30671-07ce-411e-86a7-f820f697ba6a)


![image](https://github.com/shashaktjha/hpml-project/assets/56186071/0354912a-04c6-4a05-bdf1-e3a156f0947b)

