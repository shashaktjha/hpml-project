import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch.profiler

# Ensure the log directory exists
log_dir = './logs/Quantized_Model'
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard in Colab
%load_ext tensorboard
%tensorboard --logdir {log_dir}

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp_quantized = pipeline('question-answering', model=model_quantized, tokenizer=tokenizer)

QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}

# Profile the model
def profile_model():
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU] + 
                       ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True) as prof:
        for _ in range(10):  # Increase the range if necessary to ensure activation
            nlp_quantized(QA_input)
            prof.step()  # Signal that the next step is starting

profile_model()
print("Profiling completed.")
