import json
import re
from util import chat
from random import sample, seed
import tqdm
import os

def pre_all_in_one(topic,data, model_name="qwen2.5-72b-instruct", sample_size=10, retries=10):
    """
    Generate preference descriptions for the selected topic.

    Args:
        data (list): Dataset records.
        model_name (str): Model used to generate preferences.
        sample_size (int): Number of examples to sample per dataset.
        retries (int): Maximum number of retries.

    Returns:
        list: Preference dictionaries.
    """    
    datasets = list(set([item["dataset"] for item in data]))
    # print(datasets)
    new_preference = []
    new_preference.append({"preference": ""})
    for dataset in tqdm.tqdm(datasets):
        
        if topic not in dataset:
            continue
        data_for_preference = [item for item in data if item["dataset"] == dataset]
        # print(data_for_preference)
        for dataaa in data_for_preference:
            # print(dataaa)
            prompt = f"""
You will be given a prompt and two responses: a response that was chosen by the user (Chosen Response) and a response that was rejected by the user (Rejected Response) during a pairwise comparison. 
The prompt is a "Human" utterance containing a human-assistant diaglog end with a human request or question and the responses are "Assistant" utterances that provide answers or responses for the human. Your task is to generate a very short, specific, one-sentence description of the user's persona preference, i.e. a persona. The persona preference should contain reasoning for why the user preferred and picked the Chosen Response and did not pick the Rejected Response. The persona preference should discuss higher-level characteristics that can be inferred about the user's persona. The persona preference should be very short and should not mention specific details or exact words and phrasing present in the prompt or responses. Answer in English.

Prompt: {dataaa["prompt"]}
---
Chosen Response: {dataaa["chosen"]}
---
Rejected Response:  {dataaa["rejected"]}
---
Persona Preference: 
"""
            output = chat(model_name, str(prompt))  
             
            new_preference.append({"preference": output})
        prompt = f"""
You will be given a list of data examples, each examples includes a prompt and two responses: a response that was chosen by the user (Chosen Response) and a response that was rejected by the user (Rejected Response) during a pairwise comparison. The prompt is a "Human" utterance containing a human-assistant diaglog end with a human request or question and the responses are "Assistant" utterances that provide answers or responses for the human. These data examples come from the same persona.
Your task is to generate a very short, specific, one-sentence description of the user's persona preference, i.e. a persona. The persona preference should contain reasoning for why the user preferred and picked the Chosen Response and did not pick the Rejected Response. The persona preference should discuss higher-level characteristics that can be inferred about the user's persona. The persona preference should be very short and should not mention specific details or exact words and phrasing present in the prompt or responses. Answer in English.

Data Examples: 
{data_for_preference}
---
Persona Preference:
"""
        output = chat(model_name, str(prompt))   
        new_preference.append({"preference": output})
    return new_preference


import argparse
parser = argparse.ArgumentParser(description="Process files based on provided paths.")
parser.add_argument("--dev_json_file", type=str, default="", help="")
parser.add_argument("--topic", type=str, default="", help="")
parser.add_argument("--model_name", type=str, default="deepseek-v4-pro", help="Model name used to generate preferences.")
#python /data/sense/benchmark_run/pipeline/1_get_preference.py --dev_json_file /data/sense/benchmark_run/pipeline/data/data_process/sense-annotator3-dev.json --topic all

args = parser.parse_args()

topic=args.topic

dev_json_file=args.dev_json_file

name=dev_json_file.split("/")[-1].replace(".json","")


topics=[]
if "all" not in topic:
    topics.append(topic)
else:
    topics=["Role","NLU","Writing","Logic","Translation","Math","Code","QA"]

folder_name = "data/preference"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open(dev_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for topic in topics:
    filepath=f"{folder_name}/{name}-preference"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    generated_preferences = pre_all_in_one(topic,data, model_name=args.model_name)

    with open(f"{filepath}/{topic}.json", "w", encoding="utf-8") as f:
        json.dump(generated_preferences, f, ensure_ascii=False, indent=4)
