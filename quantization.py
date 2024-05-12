import time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

model_name = "deepset/roberta-base-squad2"

# Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Set up the pipelines
nlp_original = pipeline('question-answering', model=model, tokenizer=tokenizer)
nlp_quantized = pipeline('question-answering', model=model_quantized, tokenizer=tokenizer)

QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}

# Measure inference time for the original model
start_time = time.time()
original_result = nlp_original(QA_input)
end_time = time.time()
original_inference_time = end_time - start_time

# Measure inference time for the quantized model
start_time = time.time()
quantized_result = nlp_quantized(QA_input)
end_time = time.time()
quantized_inference_time = end_time - start_time

# Print results
print("Original Model Result:", original_result)
print("Quantized Model Result:", quantized_result)
print("Original Model Inference Time: {:.6f} seconds".format(original_inference_time))
print("Quantized Model Inference Time: {:.6f} seconds".format(quantized_inference_time))
