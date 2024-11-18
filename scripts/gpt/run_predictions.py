import pandas as pd
from datasets import Dataset
from openai import OpenAI
import ast
from tqdm import tqdm
# Initialize OpenAI client
client = OpenAI(api_key="<OpenAI-key>")

def query_llm(example):
    """
    Send a prompt to the LLM and return the response.
    """
    print(example)
    response = client.chat.completions.create(
        messages=example,
        model="ft:gpt-4o-2024-08-06:personal:elevate-id-mix:ATTQbjfY"#"ft:gpt-4o-2024-08-06:personal:indel:AT4C3C4Y", #change the model to model="gpt-3.5-turbo if you want to use gpt-3.5 
    )
    return {"response": response.choices[0].message.content}

def create_input_prompt(example):
    system_message = """
    Find entities and their corresponding entry links in Wikidata within the following sentence.
    Use the context of the sentence to determine the correct entries in Wikidata.
    The output should be formatted as: [entity1=link1, entity2=link2]. 
    No explanations are needed.
    """
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example["sentence"]},
        ]
    }
def read_first_few_lines(file_path, num_lines=0):
    lines = []
    try:
        with open(file_path, 'r') as file:
            if num_lines==0:
                lines = file.readlines()
            else:
                for _ in range(num_lines):
                    lines.append(file.readline().strip())
    except Exception as e:
        return str(e)
    return lines


# Read and display the first few lines of the file
domain = "general-domain"
file_path = f"data/{domain}/test_final.jsonl"
first_few_lines = read_first_few_lines(file_path)
linking_results = []
for line in tqdm(first_few_lines):
    line = ast.literal_eval(line)
    results = query_llm(line["content"]["messages"])
    print(results)
    if len(results["response"]) == 0:
        results["response"] = []
    linking_results.append((line['sent_id'], results['response']))

f = open("results.txt", "w")
for linking_result in linking_results:
    sent_id, links = linking_result
    f.write(f"{sent_id}\t{links}\n")
f.close()