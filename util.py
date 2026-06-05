# API helpers.
import requests
import json
import time
import re
import os
from loguru import logger
from itertools import combinations


def build_chat_completions_url(base_url):
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def get_model_api_base(model_name):
    if os.environ.get("MODEL_API_BASE"):
        return os.environ["MODEL_API_BASE"]
    if os.environ.get("OPENAI_API_BASE"):
        return os.environ["OPENAI_API_BASE"]
    if os.environ.get("SELF_HOSTED_API_BASE"):
        return os.environ["SELF_HOSTED_API_BASE"]
    if model_name.startswith("deepseek"):
        base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
        if base_url.endswith("/v1") or base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/v1"
    return os.environ.get("LOCAL_MODEL_API_BASE", "http://localhost:8000/v1")


def get_model_api_key(model_name):
    if os.environ.get("MODEL_API_KEY") is not None:
        return os.environ["MODEL_API_KEY"]
    if os.environ.get("OPENAI_API_KEY") is not None:
        return os.environ["OPENAI_API_KEY"]
    if model_name.startswith("deepseek"):
        return os.environ.get("DEEPSEEK_API_KEY", "")
    return os.environ.get("LOCAL_MODEL_API_KEY", "")


def get_extra_headers():
    raw_headers = os.environ.get("MODEL_API_EXTRA_HEADERS", "")
    if not raw_headers:
        return {}
    try:
        headers = json.loads(raw_headers)
    except json.JSONDecodeError as exc:
        raise ValueError("MODEL_API_EXTRA_HEADERS must be a JSON object.") from exc
    if not isinstance(headers, dict):
        raise ValueError("MODEL_API_EXTRA_HEADERS must be a JSON object.")
    return {str(key): str(value) for key, value in headers.items()}


class ChatAgent:
    def __init__(self, model_name, max_tokens=1024, ip=None, port=None):
        self.model_name = model_name
        self.max_tokens = int(os.environ.get("MODEL_API_MAX_TOKENS", os.environ.get("DEEPSEEK_MAX_TOKENS", str(max_tokens))))
        if model_name.startswith("mock"):
            self.url = None
            self.header = {}
            return

        if ip and port:
            api_base = f"http://{ip}:{port}/v1"
        else:
            api_base = get_model_api_base(model_name)

        api_key = get_model_api_key(model_name)
        if model_name.startswith("deepseek") and not api_key:
            raise ValueError("DEEPSEEK_API_KEY, MODEL_API_KEY, or OPENAI_API_KEY is required for DeepSeek models.")

        self.url = build_chat_completions_url(api_base)
        self.header = {"Content-Type": "application/json"}
        if api_key:
            self.header["Authorization"] = f"Bearer {api_key}"
        self.header.update(get_extra_headers())

        if not self.url:
            raise ValueError("Model API base URL is empty. Set MODEL_API_BASE or SELF_HOSTED_API_BASE.")

    def chat(self, message: str, retry=5):
        if self.model_name.startswith("mock"):
            if "response-b" in self.model_name:
                return "The final decision is Response B."
            return "The final decision is Response A."

        if retry == 0:
            raise Exception("Retry failed")

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],  
            "max_tokens": int(self.max_tokens),
            "temperature": float(os.environ.get("MODEL_API_TEMPERATURE", "0")),
            "stream": False,
        }
        if os.environ.get("MODEL_API_TOP_P"):
            data["top_p"] = float(os.environ["MODEL_API_TOP_P"])
        thinking = os.environ.get("MODEL_API_THINKING", os.environ.get("DEEPSEEK_THINKING", ""))
        if thinking:
            data["thinking"] = {"type": thinking}
        reasoning_effort = os.environ.get("MODEL_API_REASONING_EFFORT", os.environ.get("DEEPSEEK_REASONING_EFFORT", ""))
        if reasoning_effort:
            data["reasoning_effort"] = reasoning_effort

        try:
            response = requests.post(
                self.url,
                headers=self.header,
                json=data,
                timeout=int(os.environ.get("MODEL_REQUEST_TIMEOUT", os.environ.get("MODEL_API_TIMEOUT", "120"))),
            )
        except requests.RequestException as exc:
            logger.warning(str(exc))
            return self.chat(message, retry-1)
        if response.status_code != 200:
            logger.warning(response.text)
            return self.chat(message, retry-1)

        res = response.json()
        if os.environ.get("VERBOSE_MODEL_OUTPUT") == "1":
            print(res["choices"][0]["message"]["content"])
        return res["choices"][0]["message"]["content"]

import json
from itertools import combinations
from datetime import datetime
import tqdm
import os

def load_and_process_data(file_path, reverse):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Error: file not found: {file_path}\n"
    except json.JSONDecodeError:
        return f"Error: failed to decode JSON file: {file_path}\n"

    output_matrix = []
    for item in data:
        if item and "outputs" in item and isinstance(item["outputs"], list):
            row = []
            # print(len(item["outputs"]))
            for output_item in item["outputs"]:
                
                if isinstance(output_item, dict) and "output" in output_item:
                    if (("最终决定是回应B" in output_item["output"] or "The final decision is Response B" in output_item["output"]) and reverse) or (("最终决定是回应A" in output_item["output"] or "The final decision is Response A" in output_item["output"]) and not reverse):
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    row.append(None)
                # print(row)
            output_matrix.append(row)
        else:
            output_matrix.append([])
    return output_matrix


def count_rows_over_half(matrix, reverse, selected_cols):
    rows_over_half = 0
    num_selected_cols = len(selected_cols)
    half_len = num_selected_cols / 2
    for row in matrix:
        num_ones = 0
        valid_count = 0
        for col_index in selected_cols:
            if col_index < len(row):
                value = row[col_index]
                if value is not None:
                    valid_count += 1
                    if value == 1:
                        num_ones += 1

        if valid_count > 0:
            if (reverse and num_ones >= valid_count / 2) or (not reverse and num_ones >= valid_count / 2):
                rows_over_half += 1
    return rows_over_half

def analyze_combinations(matrix_true, matrix_false, consider_difference):
    log_content = ""
    if isinstance(matrix_true, str) or isinstance(matrix_false, str):
        return matrix_true if isinstance(matrix_true, str) else matrix_false

    num_rows_true = len(matrix_true)
    num_rows_false = len(matrix_false)

    max_cols_true = 0
    for row in matrix_true:
        max_cols_true = max(max_cols_true, len(row))

    max_cols_false = 0
    for row in matrix_false:
        max_cols_false = max(max_cols_false, len(row))

    max_cols = max(max_cols_true, max_cols_false)
    results = {}

    for i in range(1, max_cols + 1):
        for selected_cols_tuple in combinations(range(max_cols), i):
            selected_cols = sorted(list(selected_cols_tuple))

            count_true = count_rows_over_half(matrix_true, True, selected_cols)
            count_false = count_rows_over_half(matrix_false, False, selected_cols)

            if consider_difference:
                evaluation_metric = (count_true + count_false) - abs(count_true - count_false)
            else:
                evaluation_metric = count_true + count_false
            results[tuple(selected_cols)] = evaluation_metric

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

    # analysis_type = "with difference penalty" if consider_difference else "without difference penalty"
    # log_content += f"\n{'='*30} Analysis type: {analysis_type} {'='*30}\n"
    # log_content += f"\nAll column combinations and metrics, sorted descending:\n"
    # for cols, metric in sorted_results:
    #     count_t = count_rows_over_half(matrix_true, True, list(cols))
    #     count_f = count_rows_over_half(matrix_false, False, list(cols))
    #     if consider_difference:
    #         log_content += f"columns {list(cols)}: metric = {metric} (total rows = {count_t + count_f}, difference = {abs(count_t - count_f)})\n"
    #     else:
    #         log_content += f"columns {list(cols)}: total rows = {metric}\n"

    # log_content += f"\nTop 10 column combinations by metric:\n"
    # for i, (cols, metric) in enumerate(sorted_results[:10]):
    #     count_t_best = count_rows_over_half(matrix_true, True, list(cols))
    #     count_f_best = count_rows_over_half(matrix_false, False, list(cols))
    #     if consider_difference:
    #         log_content += f"{i+1}. columns {list(cols)}: metric = {metric}, total rows = {count_t_best + count_f_best}, difference = {abs(count_t_best - count_f_best)}\n"
    #     else:
    #         log_content += f"{i+1}. columns {list(cols)}: total rows = {metric}\n"
        # print(i)
    # log_file.write(log_content)
    return sorted_results

def analyze_sorted_results(sorted_results_chazhi, sorted_results_no_new, matrix_true, matrix_false, log_content,start, count_i):

    first_elements = [pair[0] for pair in sorted_results_chazhi[start:count_i]]
    # first_elements = [pair[0] for pair in sorted_results_chazhi[start:60]]
    # print(f"Top difference columns: {first_elements}")
    # print(first_elements)
    # json.dump(first_elements,open("/home/liujunfeng/best_combinnation.json","w"),ensure_ascii=False,indent=4)
    metrics = []
    count_t_bests=[]
    count_f_bests=[]
    # print(sorted_results_no_new)
    for i, (cols, metric) in enumerate(sorted_results_no_new):
        if cols not in first_elements:
            continue
        count_t_best = count_rows_over_half(matrix_true, True, list(cols))
        count_f_best = count_rows_over_half(matrix_false, False, list(cols))

        count_t_bests.append(count_t_best)
        count_f_bests.append(count_f_best)

        metrics.append(metric)
        log_content += f"{i+1}. columns {list(cols)}: total rows = {metric}\n"

    if metrics:
        avg_5 = sum(metrics) / len(metrics)
        print(sum(metrics))
        print(len(metrics))
        avg_t_5=sum(count_t_bests) / len(count_t_bests)
        avg_f_5=sum(count_f_bests) / len(count_f_bests)
    return log_content, avg_5,avg_t_5,avg_f_5


_chat_agents = {}


def chat(model_name, message, max_tokens=512):
    max_tokens = int(os.environ.get("MODEL_API_MAX_TOKENS", os.environ.get("DEEPSEEK_MAX_TOKENS", str(max_tokens))))
    agent_key = (model_name, max_tokens)
    if agent_key not in _chat_agents:
        _chat_agents[agent_key] = ChatAgent(model_name, max_tokens=max_tokens)
    return _chat_agents[agent_key].chat(message)
