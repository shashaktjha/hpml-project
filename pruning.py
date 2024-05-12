import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import time
from torch.nn.utils import prune

model_name = "deepset/roberta-base-squad2"

# Load the pre-trained model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline with the original model
nlp_original = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Apply pruning to the model
# Here we are pruning 30% of the connections in the first linear layer of the model
parameters_to_prune = (
    (model.roberta.encoder.layer[0].intermediate.dense, 'weight'),
    (model.roberta.encoder.layer[0].output.dense, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.9,
)

# To make the pruning permanent
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Create a pipeline with the pruned model
nlp_pruned = pipeline('question-answering', model=model, tokenizer=tokenizer)

QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and lets people easily switch between frameworks.'
}

# Measure inference time for the original model
start_time = time.time()
original_result = nlp_original(QA_input)
original_inference_time = time.time() - start_time

# Measure inference time for the pruned model
start_time = time.time()
pruned_result = nlp_pruned(QA_input)
pruned_inference_time = time.time() - start_time

# Print results and comparison
print("Original Model Inference Time: {:.6f} seconds".format(original_inference_time))
print("Pruned Model Inference Time: {:.6f} seconds".format(pruned_inference_time))
print("Original Model Answer:", original_result['answer'])
print("Pruned Model Answer:", pruned_result['answer'])
