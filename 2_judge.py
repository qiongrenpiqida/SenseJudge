import os
import re
import json
import tqdm 
import argparse
import requests 

from datetime import datetime

from loguru import logger 
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__) 
count_all = 0
count_no = 0
import datetime

now_time = datetime.datetime.now().time()

from itertools import permutations, combinations
import tiktoken

from util import ChatAgent
from analysis import load_and_process_data,count_rows_over_half,analyze_combinations,select_combinations_by_dev_strategy

def extract_last_number(text):
  """
  Extract the last trailing number from text.
  """
  match = re.search(r'\d+$', text)
  if match:
    return match.group(0)
  else:
    return None

def count_tokens_openai(text, model_name="cl100k_base"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def count_final_decision(text,target_string):

  target_string = "The final decision is Response A"
  count = text.count(target_string)
  return count


def normalize_preference(item_pre):
    if isinstance(item_pre, dict):
        return str(item_pre.get("preference", ""))
    return str(item_pre or "")



# Set up argument parser
parser = argparse.ArgumentParser(description="Save counts to a file.")
parser.add_argument("--model_name", type=str, required=True, help="Count for 'chuangzuo' datasets")
parser.add_argument("--topic", type=str, required=True, help="Count for 'chuangzuo' datasets")

parser.add_argument("--preference_path", type=str, required=True, help="Count for 'chuangzuo' datasets")

parser.add_argument("--num_workers", default=16, type=int, help="Number of samples to parallel. When use proxy model, set it to less than 8.")

parser.add_argument("--data_path", default="/home/lirui/math-shield/judger/dirachat_arena_sensebmk_1217.json", type=str, help="dataset path (json format)")

parser.add_argument("--file_path_false", type=str, default="test_mode", help="Path to the 'false' file.")
parser.add_argument("--max_samples", default=0, type=int, help="Limit matched samples per topic. 0 means no limit.")


def run_evaluation(preference_chosen,name,data, model, meta_prompt, num_workers,topic,preference_path, output_dir=".",reverse=False, max_samples=0):
    new_data=[]
    data_pref = json.load(open(preference_path, "r"))
    matched_data = [item for item in data if topic in item["dataset"]]
    if max_samples > 0:
        matched_data = matched_data[:max_samples]

    def eval_sample(item):
        query = str(item["prompt"])
        chosen = str(item["chosen"]["value"][0]["value"])
        rejected = str(item["rejected"]["value"][0]["value"])
        if "</think>" in chosen:
            chosen=chosen.split("</think>")[-1]
        if "</think>" in rejected:
            rejected=rejected.split("</think>")[-1]

        count_token=count_tokens_openai(query+chosen+rejected)

        # print(count_token)
        if  count_token>10000:
            print("Original token count")
            print(count_token)
            query = str(item["prompt"][-1])
            
            count_token=count_tokens_openai(query)
            print("Recomputed token count")
            print(count_token)
            
        # query = item["prompt"]
        # chosen = item["chosen"]["value"][0]["value"]
        # rejected = item["rejected"]["value"][0]["value"]
        outputs=[]
        preference=""
        count_preference=0
        for item_pre in data_pref:
            preference=normalize_preference(item_pre)
            if count_preference in preference_chosen:
                if reverse:
                    prompt = meta_prompt.format(query, rejected,chosen,preference)
                else:
                    prompt = meta_prompt.format(query, chosen, rejected,preference)
                output = model.chat(prompt)
                outputs.append({"output":output,"preference":preference})
            else:
                outputs.append({"output":"","preference":preference})
            count_preference += 1
        temp = {
            "query": query,
            "chosen": chosen,
            "rejected": rejected,
            "outputs": outputs,
            "dataset": item["dataset"]
        }
        return temp

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        new_data = list(tqdm.tqdm(executor.map(eval_sample, matched_data), total=len(matched_data)))
    preference_path_get=preference_path.split('/')[-2].replace(".json",'')
    # output_dir_path = os.path.join(output_dir, f"{model.model_name}_{preference_path_get}")
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.join(output_dir, f"{topic}_output_reverse_{reverse}.json")

    new_data = [item for item in new_data if item is not None]
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print("--------------finished--------------")


meta_prompt="""
You are going to evaluate two responses to a given user query and determine which response is superior. Below is the relevant content:

[BEGIN DATA]
***
[User Query]:{}
***
[Response A]: {}
***
[Response B]: {}
***
[END DATA]

Here are the guidelines for evaluating and comparing the two responses:

##########BEGIN User Preferences##########
{}
##########END User Preferences##########

1. Score each of the two responses based on the user preferences.
2. Based on the scores obtained in the first step, determine which response is better. If Response A is better, output "The final decision is Response A." If Response B is better, output "The final decision is Response B."
""".strip()




if __name__ == "__main__":
    args = parser.parse_args()
    model = ChatAgent(args.model_name)
    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    name=args.data_path.split('/')[-1].replace(".json",'')
    file_path_false = args.file_path_false


    folder_name = "data/result"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    topic=args.topic
    if "test_mode" not in file_path_false:
        file_path_true=file_path_false.replace("False","True")

        matrix_true = load_and_process_data(file_path_true, True)
        matrix_false = load_and_process_data(file_path_false, False)
        sorted_results, strategy, forward_best, inverted_worst = select_combinations_by_dev_strategy(
            topic,
            matrix_true,
            matrix_false,
            False,
        )
        if os.environ.get("VERBOSE_COMBINATIONS") == "1":
            print(sorted_results)
            print(strategy, forward_best, inverted_worst)
            print(list(sorted_results[0][0]))
        preference_chosen=list(sorted_results[0][0])
        preference_chosen.append(0)
        best_metric=sorted_results[0][1]
        for cols, metric in sorted_results:
            if metric==best_metric:
                preference_chosen+=list(cols)
    else:
        preference_chosen=[0,1,2,3,4,5,6,7,8,9,10,11]

    if os.environ.get("VERBOSE_COMBINATIONS") == "1":
        print(preference_chosen)
    else:
        print(f"Selected preference index count: {len(preference_chosen)}")
    output_dir=f"data/result/{name}-{args.model_name}"
    file_name = os.path.join(output_dir, f"{topic}_output_reverse_False.json")
    if not os.path.exists(file_name):
        run_evaluation(preference_chosen,name,data, model, meta_prompt, args.num_workers,preference_path=args.preference_path,topic=args.topic, output_dir=f"data/result/{name}-{args.model_name}",reverse=False, max_samples=args.max_samples)
    output_dir=f"data/result/{name}-{args.model_name}"
    
    file_name = os.path.join(output_dir, f"{topic}_output_reverse_True.json")
    if not os.path.exists(file_name):
        run_evaluation(preference_chosen,name, data, model, meta_prompt, args.num_workers, preference_path=args.preference_path,topic=args.topic, output_dir=f"data/result/{name}-{args.model_name}",reverse=True, max_samples=args.max_samples)
