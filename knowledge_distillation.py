import torch
from transformers import RobertaForQuestionAnswering, RobertaTokenizer, RobertaConfig
from datasets import load_dataset
from transformers import pipeline
import time

# Load the full teacher model
teacher_model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
teacher_model.eval()

# Setup the student model with a reduced configuration
student_config = RobertaConfig.from_pretrained('deepset/roberta-base-squad2', num_hidden_layers=6)  # fewer layers
student_model = RobertaForQuestionAnswering(student_config)
student_model.eval()

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')

# Assuming student model is already trained and loaded for inference
# In practice, you need to train it using a distillation approach as shown in previous messages
# Example QA input
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}

def measure_performance(model, QA_input):
    # Setup pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Measure inference time
    start_time = time.time()
    result = nlp(QA_input)
    end_time = time.time()
    inference_time = end_time - start_time

    return result, inference_time

# Measure performance for both teacher and student models
teacher_result, teacher_time = measure_performance(teacher_model, QA_input)
student_result, student_time = measure_performance(student_model, QA_input)

# Print comparison
print("Teacher Model Result:", teacher_result)
print("Teacher Model Inference Time: {:.6f} seconds".format(teacher_time))
print("Student Model Result:", student_result)
print("Student Model Inference Time: {:.6f} seconds".format(student_time))
