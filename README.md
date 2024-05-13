This project explores the application of knowledge distillation and model compression techniques to optimize deep learning models, specifically focusing on a question answering task using the RoBERTa model from the Hugging Face transformers library. The aim is to reduce model size and inference time while maintaining or improving model accuracy and performance.

Project Milestones
Milestone 1: Implement Knowledge Distillation
Status: Completed
Description: Implemented a knowledge distillation technique where a smaller student model learns from a larger teacher model.
Milestone 2: Implement Model Compression
Status: Completed
Description: Applied model compression techniques, including pruning and quantization, to reduce the size and increase the inference speed of the RoBERTa model.
Milestone 3: Setup and Evaluate Distributed Deep Learning
Status: Completed
Description: Configured and evaluated distributed deep learning to leverage multiple GPUs, enhancing training efficiency.
Milestone 4: Performance Evaluation and Comparison
Status: Completed
Description: Assessed the performance impact of knowledge distillation and model compression through various metrics such as inference time and accuracy.
Repository and Code Structure
Knowledge_Distillation.py: Contains the implementation of knowledge distillation with examples of teacher and student model setups.
model_compression_ddl.py: Includes the application of model compression techniques and distributed deep learning configurations.
profiling.py: Script for profiling model performance, especially focusing on quantized models.
pruning.py: Demonstrates the application of pruning techniques to the model.
quantization.py: Shows how dynamic quantization is applied to the model.
bash.sh: Batch script for running the models on the NYU HPC cluster.

How to Run the Code
To execute the scripts on the NYU HPC cluster, follow these steps:
-Upload the code to nyu hpc with the batch file
- run sbatch bash.sh




Results and Observations:

![image](https://github.com/shashaktjha/hpml-project/assets/56186071/04a95518-077c-447c-a8b1-a2aafc6635dc)

![image](https://github.com/shashaktjha/hpml-project/assets/56186071/fad30671-07ce-411e-86a7-f820f697ba6a)


![image](https://github.com/shashaktjha/hpml-project/assets/56186071/0354912a-04c6-4a05-bdf1-e3a156f0947b)

