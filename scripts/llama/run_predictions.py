from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import torch
import re
import os
import pandas as pd
from datasets import Dataset

def extract_first_assistant_values(output):
    # Remove content within tags
    cleaned_text = re.sub(r'<[^>]*>', '', output)

    # Find the assistant's response after the word "assistant"
    response = re.search(r"assistant\s*\n\s*(.*)", cleaned_text, re.DOTALL)
    if response:
        links = response.group(1).split(', ')
        unique_links = []
        seen_entities = set()
        for link in links:
            if "=" not in link:
                continue
            entity = link.split("=")[0]
            if entity not in seen_entities:
                seen_entities.add(entity)
                unique_links.append(link)
        first_distinct_answer = ', '.join(unique_links)
        return first_distinct_answer
    return []
    
def generate_answer(example):
    
    prompt = pipe.tokenizer.apply_chat_template(example["messages"][:2],
                                                tokenize=False,
                                                add_generation_prompt=True)
    terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
    
    outputs = pipe(prompt,
                max_new_tokens=100,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.3
                )
    generated_text = outputs[0]['generated_text'][len(prompt):]
    return {"sent_id": example["sent_id"], "sentence": example["messages"][1]['content'], "generated_text": generated_text}

def create_input_prompt(example):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": example["sentence"]},
        ]
    }

# setting-parameters
domain = "general-domain"
base_model_name = "fine-tuning-results/mix-domain/meta-llama/Meta-Llama-3-70B-Instruct/checkpoint-1404"
#base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# Read CSV files
test_df = pd.read_csv(f"../../datasets/{domain}/test_set.txt")

# Convert DataFrame to Dataset
test_dataset = Dataset.from_pandas(test_df)

system_message = """
Find entities and their corresponding entry links in Wikidata within the following sentence.
Use the context of the sentence to determine the correct entries in Wikidata.
The output should be formatted as: [entity1=link1, entity2=link2]. 
No explanations are needed.
"""

# Load Model with PEFT adapter
model = AutoModelForCausalLM.from_pretrained(
  base_model_name,
  device_map="auto",
  torch_dtype=torch.float16,
  offload_buffers=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Apply the prepare_examples function
test_dataset = test_dataset.map(create_input_prompt, remove_columns=['sentence', 'entities', 'uris'])

# Generate answers for the dataset
results = test_dataset.map(generate_answer, batched=False)

# Check data dir and create it if does not exist
directory_data = 'data'
if not os.path.exists(directory_data):
    os.makedirs(directory_data)
if not os.path.exists(f"{directory_data}/{domain}"):
    os.makedirs(f"{directory_data}/{domain}")
if not os.path.exists(f"{directory_data}/{domain}/{base_model_name.split('/')[-1]}"):
    os.makedirs(f"{directory_data}/{domain}/{base_model_name.split('/')[-1]}")
    
f = open(f"data/{domain}/{base_model_name.split('/')[-1]}/results.txt", "w")
for result in results:
    # Extract values from the output
    values = result['generated_text'] #extract_first_assistant_values(result['generated_text'])
    print(result['sent_id'], result['sentence'])
    print(result)
    print("")
    f.write(f"{result['sent_id']}\t{values}\n")
f.close()