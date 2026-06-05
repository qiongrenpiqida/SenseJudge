import json
import random
import os
import tqdm

def response_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value", "")
        if isinstance(nested, list):
            parts = []
            for item in nested:
                if isinstance(item, dict) and "value" in item:
                    parts.append(str(item["value"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(nested)
    return str(value)


def deduplicate_unordered_pairs(data):
    deduped = []
    seen = set()
    for item in data:
        response_a = response_text(item.get("chosen", ""))
        response_b = response_text(item.get("rejected", ""))
        left, right = sorted([response_a, response_b])
        key = (
            str(item.get("prompt", item.get("query", ""))),
            str(item.get("dataset", "")),
            left,
            right,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def split_json_data(input_file, dev_file="dev.json", test_file="test.json", num_samples=10, seed=None):
    """
    Randomly sample a fixed number of records per topic as the dev set.
    The remaining records are written to the test set.

    Args:
        input_file (str): Input JSON path.
        dev_file (str): Output dev JSON path.
        test_file (str): Output test JSON path.
        num_samples (int): Number of dev samples per topic.
        seed (int, optional): Random seed for reproducibility.
    """
    dev_dataset=[]
    test_dataset=[]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    original_count = len(data)
    data = deduplicate_unordered_pairs(data)
    removed_count = original_count - len(data)
    if removed_count:
        print(f"Removed {removed_count} unordered chosen/rejected duplicate samples; {len(data)} samples remain.")
    topics = list(set([item["dataset"] for item in data]))
    for topic in tqdm.tqdm(topics):
        subdataset = [item for item in data if item["dataset"] == topic]
        if seed is not None:
            random.seed(seed)

        if len(subdataset) < num_samples:
            print(
                f"Warning: topic has only {len(subdataset)} samples, fewer than requested "
                f"{num_samples}. All samples will be used as dev and the test split will be empty."
            )
            dev_subdataset = subdataset
            test_subdataset = []
        else:
            # Randomly select dev indices.
            random_indices = random.sample(range(len(subdataset)), num_samples)
            dev_subdataset = [subdataset[i] for i in random_indices]
            test_subdataset = [subdataset[i] for i in range(len(subdataset)) if i not in random_indices]
            dev_dataset.extend(dev_subdataset)
            test_dataset.extend(test_subdataset)
    # Save dev data.
    try:
        with open(dev_file, 'w', encoding='utf-8') as f:
            json.dump(dev_dataset, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(dev_dataset)} dev samples to '{dev_file}'.")
    except IOError:
        print(f"Error: failed to write '{dev_file}'.")

    # Save test data.
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(test_dataset)} test samples to '{test_file}'.")
    except IOError:
        print(f"Error: failed to write '{test_file}'.")


import argparse
parser = argparse.ArgumentParser(description="Process files based on provided paths.")
parser.add_argument("--input_json_file", type=str, default="", help="")
args = parser.parse_args()

if __name__ == "__main__":
    data_path="data"
    input_json_file = args.input_json_file
    num_dev_samples = 10
    folder_name = "data/data_process"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    name=input_json_file.split("/")[-1].replace(".json","")
    dev_file=f"{folder_name}/{name}-dev.json"
    test_file=f"{folder_name}/{name}-test.json"
    split_json_data(input_json_file,dev_file=dev_file, test_file=test_file,num_samples=num_dev_samples, seed=42)

# python /data/sense/benchmark_run/pipeline/0_get_dev_for_preference.py --input_json_file /data/sense/benchmark_run/pipeline/data/sense-annotator3.json
