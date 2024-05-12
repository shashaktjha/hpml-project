import os
import torch
import torch.distributed as dist
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import F1Score
import time

def setup_distributed_environment(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare_data(tokenizer):
    dataset = load_dataset('squad')
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Dataset does not contain the required splits: 'train' and 'validation'")
    
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples['question'], 
            examples['context'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        ), batched=True
    )
    
    train_size = len(tokenized_datasets['train'])
    valid_size = len(tokenized_datasets['validation'])
    print(f"Size of training set: {train_size}")
    print(f"Size of validation set: {valid_size}")

    if train_size == 0 or valid_size == 0:
        raise ValueError("One of the dataset splits is empty after processing.")

    return tokenized_datasets

def train_model(model, tokenized_datasets, args, tokenizer):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer
    )
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f}s")
    return trainer, training_time

def evaluate_model(model, tokenized_datasets, calculate_inference_time=False):
    f1 = F1Score()
    model.eval()
    inference_start_time = time.time()
    batches_processed = 0
    
    for i, batch in enumerate(tokenized_datasets['validation']):
        if batch['input_ids'].size(0) == 0:
            print(f"Warning: Empty batch at index {i}")
            continue
        batches_processed += 1
        inputs = {'input_ids': batch['input_ids'].to(model.device), 'attention_mask': batch['attention_mask'].to(model.device)}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        f1.update(predictions, batch['labels'].to(model.device))
    
    if batches_processed == 0:
        raise RuntimeError("No valid batches were processed during evaluation.")
    
    f1_score = f1.compute()
    inference_time = time.time() - inference_start_time
    if calculate_inference_time:
        print(f"Inference Time: {inference_time:.2f}s")
    return f1_score, inference_time

def main(rank, world_size, use_ddp, use_compression):
    if use_ddp:
        setup_distributed_environment(rank, world_size)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(rank)
    if use_ddp:
        model = DDP(model, device_ids=[rank])
    tokenized_datasets = prepare_data(tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to='none'
    )

    trainer, training_time = train_model(model, tokenized_datasets, training_args, tokenizer)
    f1_score, inference_time = evaluate_model(trainer.model, tokenized_datasets, calculate_inference_time=True)

    if use_compression:
        model = torch.quantization.quantize_dynamic(model.module if hasattr(model, "module") else model, {torch.nn.Linear}, dtype=torch.qint8)
        f1_score_compressed, inference_time_compressed = evaluate_model(model, tokenized_datasets, calculate_inference_time=True)
        print(f"F1 Score After Compression: {f1_score_compressed}")
    else:
        print(f"F1 Score: {f1_score}")

    if use_ddp:
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    scenarios = [(True, True), (True, False), (False, True)]
    for use_ddp, use_compression in scenarios:
        description = "Distributed + Compression" if use_ddp and use_compression else "Only Distributed" if use_ddp else "Only Compression"
        print(f"Running scenario: {description}")
        torch.multiprocessing.spawn(main, args=(world_size, use_ddp, use_compression), nprocs=world_size)
