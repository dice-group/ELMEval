import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

import warnings
import os
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import ast

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="The `use_auth_token` argument is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="`--push_to_hub_token` is deprecated")

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Setting parameters
domain = "specific-domain"
base_model_name = "meta-llama/Meta-Llama-3-70B-Instruct" # other models: Ichsan2895/Merak-7B-v4

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, max_length=512)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Read CSV files
train_df = pd.read_csv(f"../../datasets/{domain}/train_set.txt")
val_df = pd.read_csv(f"../../datasets/{domain}/val_set.txt")

print("Train DataFrame:")
print(train_df.head())

print("\nValidation DataFrame:")
print(val_df.head())

system_message = """
Find entities and their corresponding entry links in Wikidata within the following sentence.
Use the context of the sentence to determine the correct entries in Wikidata.
The output should be formatted as: [entity1=link1, entity2=link2]. 
No explanations are needed.
"""

# Define the reasoning and entity linking task
def prepare_examples(example):
    reasoning = ", ".join(f"{entity}={uri}" for entity, uri in zip(ast.literal_eval(example['entities']), ast.literal_eval(example['uris'])))
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": example["sentence"]},
            {"role": "assistant", "content": reasoning}
        ]
    }

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Apply the prepare_examples function
train_dataset = train_dataset.map(prepare_examples, remove_columns=['sentence', 'entities', 'uris'])
val_dataset = val_dataset.map(prepare_examples, remove_columns=['sentence', 'entities', 'uris'])

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_dir = f"./fine-tuning-results/{domain}/{base_model_name}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
device_map = "auto"#{"": "cuda"}
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Increased batch size
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,  # Enable mixed precision training
    bf16=False,
    max_grad_norm=0.3,
    logging_steps=10,
    warmup_ratio=0.03,
    warmup_steps=100,
    group_by_length=True,
    lr_scheduler_type="constant",
    dataloader_num_workers=4,  # Use multiple workers for data loading
    push_to_hub=False
)

trainer = SFTTrainer(
    model=base_model,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args, max_seq_length=512  # Directly pass max_seq_length here if needed
)

trainer.train()
trainer.model.save_pretrained(output_dir)