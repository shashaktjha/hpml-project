import os
import torch
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn import DataParallel
import torch.nn.utils.prune as prune
import time
from transformers import default_data_collator

def prepare_data(tokenizer):
    dataset = load_dataset('squad', split={'train': 'train[:10%]', 'validation': 'validation[:10%]'})

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['question'],
            examples['context'],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=True
        )
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(tokenized_inputs['offset_mapping']):
            answer = examples['answers'][i]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            start_positions.append(next((idx for idx, (o_start, o_end) in enumerate(offsets) if o_start <= start_char < o_end), -1))
            end_positions.append(next((idx for idx, (o_start, o_end) in enumerate(offsets) if o_start < end_char <= o_end), -1))

        tokenized_inputs.update({'start_positions': start_positions, 'end_positions': end_positions})
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

    return tokenized_datasets

def apply_pruning(model):
    # Pruning 20% of connections in the linear layers
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
    return model

def train_model(model, tokenized_datasets, args, tokenizer):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f}s")
    return trainer

def evaluate_model(model, tokenized_datasets):
    model.eval()
    correct = 0
    total = 0
    total_inference_time = 0

    for batch in tokenized_datasets['validation']:
        inputs = {
            'input_ids': batch['input_ids'].to(model.device),
            'attention_mask': batch['attention_mask'].to(model.device)
        }
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0) if inputs['input_ids'].ndimension() == 1 else inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0) if inputs['attention_mask'].ndimension() == 1 else inputs['attention_mask']

        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
        total_inference_time += inference_time

        start_predictions = torch.argmax(outputs.start_logits, dim=1)
        end_predictions = torch.argmax(outputs.end_logits, dim=1)

        correct += ((start_predictions == batch['start_positions']) & (end_predictions == batch['end_positions'])).sum().item()
        total += inputs['input_ids'].size(0)

    accuracy = correct / total
    average_inference_time = total_inference_time / total
    print(f"Evaluation Accuracy: {accuracy:.2f}")
    print(f"Average Inference Time per Batch: {average_inference_time:.4f} seconds")
    return accuracy

def main(use_ddp, use_compression):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenized_datasets = prepare_data(tokenizer)

    model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
    if torch.cuda.device_count() > 1 and use_ddp:
        model = DataParallel(model).cuda()  # Apply DataParallel to use multiple GPUs

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=16,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to='none',
        remove_unused_columns=False,
        num_train_epochs=3
    )

    if use_compression:
        model = apply_pruning(model)

    trainer = train_model(model, tokenized_datasets, training_args, tokenizer)
    evaluate_model(trainer.model, tokenized_datasets)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Adjust based on your GPU availability
    scenarios = [
        (True, True)     # Both DataParallel and model compression (pruning)
    ]
    for use_ddp, use_compression in scenarios:
        description = "Distributed + Compression" if use_ddp and use_compression else "Only Distributed" if use_ddp else "Only Compression" if use_compression else "Baseline"
        print(f"Running scenario: {description}")
        main(use_ddp, use_compression)
