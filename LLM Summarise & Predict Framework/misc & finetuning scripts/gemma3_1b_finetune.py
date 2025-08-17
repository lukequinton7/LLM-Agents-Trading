import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM
import os
import gc
from trl import DataCollatorForCompletionOnlyLM

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# instruction tuning with short outputs like -1, 0, +1

# Configuration 
MODEL_ID = "google/gemma-3-1b-it"
DATASET_PATH = "finetune_data.csv"
OUTPUT_DIR = "./gemma3-finetuned-qlora"
MAX_LENGTH = 2000  # 1800-2000 range- model takes up too much vram otherwise

# Load Data- 
print("Loading dataset...")
try:
    dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")
    print(f"Dataset loaded: {len(dataset)} examples")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    exit()

# Check class distribution
print("\nAnalyzing class distribution...")
sentiment_counts = {}
for example in dataset:
    sentiment = str(example["sentiment"])
    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
print(f"Class distribution: {sentiment_counts}")

# QLoRA Configuration (optimised for 16GB VRAM)
print("\nConfiguring QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# LoRA configuration with conservative settings for memory
lora_config = LoraConfig(
    r=8,  # or 16- more mem
    lora_alpha=16,  # or 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Model and Tokenizer
print("\nLoading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model with memory optimisation
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

#Prepare Model for PEFT
print("\nPreparing model for PEFT...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("\nModel architecture with LoRA adapters:")
model.print_trainable_parameters()

# clear cache before training
gc.collect()
torch.cuda.empty_cache()

# Format and Tokenize
def format_with_chat_template(examples):
    """
    Applies the Gemma chat template to format the dataset.
    """
    formatted_texts = []
    
    for prompt, sentiment in zip(examples["prompt"], examples["sentiment"]):
        # Convert sentiment to string format
        if str(sentiment) == "1":
            sentiment_output = "+1"
        elif str(sentiment) == "0":
            sentiment_output = "0"
        elif str(sentiment) == "-1":
            sentiment_output = "-1"
        else:
            sentiment_output = str(sentiment)

        # Create message structure for Gemma
        messages = [
            {"role": "user", "content": prompt},
            {"role": "model", "content": sentiment_output}
        ]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(formatted_text)

    # Tokenize with proper settings
    return tokenizer(
        formatted_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        add_special_tokens=False  # critical: template already added them
    )

print("\nFormatting and tokenizing dataset...")
tokenized_dataset = dataset.map(
    format_with_chat_template,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

# filter out any examples that might be too short
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x['input_ids']) > 10,
    desc="Filtering short sequences"
)

print(f"Tokenized dataset size: {len(tokenized_dataset)} examples")

# checks
print("\n--- Verifying Data Format ---")
if len(tokenized_dataset) > 0:
    # Decode first example to verify format
    decoded_example = tokenizer.decode(tokenized_dataset[0]['input_ids'], skip_special_tokens=False)
    print("First example (truncated):")
    print(decoded_example[:500] + "..." if len(decoded_example) > 500 else decoded_example)
    
    # Check for Gemma special tokens
    if "<start_of_turn>" in decoded_example and "<end_of_turn>" in decoded_example:
        print("Gemma special tokens found")
    else:
        print("Gemma special tokens not found")
        
    # Print token count for first example
    print(f"\nFirst example token count: {len(tokenized_dataset[0]['input_ids'])}")

# Training args
print("\n\nSetting up training arguments")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    
    # Batch size settings for 16GB VRAM
    per_device_train_batch_size=1,  # Conservative for memory
    gradient_accumulation_steps=4,   # Effective batch size = 4
    
    # Memory optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim="paged_adamw_8bit",  # More memory efficient than 32bit
    
    # Training settings
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=25,
    save_steps=200,
    eval_steps=200,
    
    # Precision and performance
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    
    # Scheduler
    lr_scheduler_type="cosine",
    
    # Other settings
    group_by_length=True,
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to="none",
    remove_unused_columns=False,
    
    # Disable evaluation during training to save memory
    eval_strategy="no",
    save_strategy="steps",
    
    # Set seed for reproducibility
    seed=42,
)

# collate
# IMPORTANT: We use DataCollatorForCompletionOnlyLM because we only want to train
# on the model's response (-1, 0, +1), not on the entire prompt.
# This is much more efficient for instruction tuning with short outputs.


# Find the response template tokens
response_template = "<start_of_turn>model"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

print(f"\nUsing DataCollatorForCompletionOnlyLM with response template: '{response_template}'")

# initialise trainer
print("\nInitializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train

try:
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Train
    trainer.train()
    
    print("complete")
    
except torch.cuda.OutOfMemoryError:
    print("cuda mem issue")
    raise
except Exception as e:
    print(f"training failed with error: {e}")
    raise

# save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
