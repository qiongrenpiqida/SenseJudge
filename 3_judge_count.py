import json
import re
from itertools import combinations
from datetime import datetime
import tqdm
import os
import random

FINAL_DECISION_PATTERNS = [
    re.compile(r"The\s*final\s*decision\s*is\s*(?:\*\*)?\s*(?:Response\s*)?([AB])", re.I),
    re.compile(r"final\s*decision\s*[:：]?\s*(?:\*\*)?\s*(?:Response\s*)?([AB])", re.I),
    re.compile(r"最终(?:决定|决策|选择|答案|回答)(?:是|为|：|:)?\s*(?:\*\*)?\s*(?:回应|回复|响应|回答|Response)?\s*([ABＡＢ])", re.I),
]


def extract_final_decision(text):
    hits = []
    compact_text = re.sub(r"\s+", "", text)
    for pattern in FINAL_DECISION_PATTERNS:
        for match in pattern.finditer(text):
            decision = match.group(1).upper().replace("Ａ", "A").replace("Ｂ", "B")
            hits.append((match.start(), decision))
        for match in pattern.finditer(compact_text):
            decision = match.group(1).upper().replace("Ａ", "A").replace("Ｂ", "B")
            hits.append((match.start(), decision))

    if not hits:
        return None
    return sorted(hits, key=lambda item: item[0])[-1][1]


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
                    decision = extract_final_decision(output_item["output"])
                    if decision is None:
                        row.append(0)
                    elif (decision == "B" and reverse) or (decision == "A" and not reverse):
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
    for row in matrix:
        is_correct, _, has_valid = row_vote(row, reverse, selected_cols)
        if has_valid and is_correct:
            rows_over_half += 1
    return rows_over_half


def row_vote(row, reverse, selected_cols):
    num_ones = 0
    valid_count = 0
    for col_index in selected_cols:
        if col_index < len(row):
            value = row[col_index]
            if value is not None:
                valid_count += 1
                if value == 1:
                    num_ones += 1

    if valid_count == 0:
        return False, False, False

    is_tie = num_ones * 2 == valid_count
    is_correct = num_ones > valid_count / 2
    if is_tie:
        # Tie means defaulting to Response A. Response A is correct only
        # when reverse is False.
        is_correct = not reverse
    return is_correct, is_tie, True


def count_paired_rows(matrix_true, matrix_false, selected_cols):
    rows_over_half = 0
    max_rows = max(len(matrix_true), len(matrix_false))
    for row_index in range(max_rows):
        if row_index < len(matrix_true):
            true_correct, true_tie, true_valid = row_vote(matrix_true[row_index], True, selected_cols)
        else:
            true_correct, true_tie, true_valid = False, False, False

        if row_index < len(matrix_false):
            false_correct, false_tie, false_valid = row_vote(matrix_false[row_index], False, selected_cols)
        else:
            false_correct, false_tie, false_valid = False, False, False

        rows_over_half += int(true_correct) + int(false_correct)
    return rows_over_half


def count_paired_denominator(matrix_true, matrix_false, selected_cols):
    denominator = 0
    max_rows = max(len(matrix_true), len(matrix_false))
    for row_index in range(max_rows):
        if row_index < len(matrix_true):
            _, _, true_valid = row_vote(matrix_true[row_index], True, selected_cols)
            denominator += int(true_valid)
        if row_index < len(matrix_false):
            _, _, false_valid = row_vote(matrix_false[row_index], False, selected_cols)
            denominator += int(false_valid)
    return denominator


def select_combinations_by_dev_strategy(topic, matrix_true, matrix_false, consider_difference=False):
    if isinstance(matrix_true, str) or isinstance(matrix_false, str):
        return matrix_true if isinstance(matrix_true, str) else matrix_false

    max_cols_true = 0
    for row in matrix_true:
        max_cols_true = max(max_cols_true, len(row))

    max_cols_false = 0
    for row in matrix_false:
        max_cols_false = max(max_cols_false, len(row))

    max_cols = max(max_cols_true, max_cols_false)
    scored_results = []

    for i in range(1, max_cols + 1):
        for selected_cols_tuple in combinations(range(max_cols), i):
            selected_cols = sorted(list(selected_cols_tuple))
            count_true = count_rows_over_half(matrix_true, True, selected_cols)
            count_false = count_rows_over_half(matrix_false, False, selected_cols)
            forward_score = count_paired_rows(matrix_true, matrix_false, selected_cols)
            denominator = count_paired_denominator(matrix_true, matrix_false, selected_cols)

            if consider_difference:
                metric = forward_score - abs(count_true - count_false)
            else:
                metric = forward_score
            scored_results.append((tuple(selected_cols), metric, forward_score, denominator))

    if not scored_results:
        return [], "forward_best", 0, 0

    best_metric = max(item[1] for item in scored_results)
    worst_forward_score = min(item[2] for item in scored_results)
    denominator = max(item[3] for item in scored_results)
    inverted_worst_metric = denominator - worst_forward_score

    if inverted_worst_metric > best_metric:
        selected = [
            (cols, inverted_worst_metric)
            for cols, _, forward_score, _ in scored_results
            if forward_score == worst_forward_score
        ]
        return selected, "inverted_worst", best_metric, inverted_worst_metric

    selected = [
        (cols, metric)
        for cols, metric, _, _ in scored_results
        if metric == best_metric
    ]
    return selected, "forward_best", best_metric, inverted_worst_metric

def analyze_combinations(topic,matrix_true, matrix_false, consider_difference):
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
            paired_count = count_paired_rows(matrix_true, matrix_false, selected_cols)

            if consider_difference:
                evaluation_metric = paired_count - abs(count_true - count_false)
            else:
                evaluation_metric = paired_count
            results[tuple(selected_cols)] = evaluation_metric

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

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

    for i, (cols, metric) in enumerate(sorted_results_no_new):
        if cols not in first_elements:
            continue
        count_t_best = count_rows_over_half(matrix_true, True, list(cols))
        count_f_best = count_rows_over_half(matrix_false, False, list(cols))

        count_t_bests.append(count_t_best)
        count_f_bests.append(count_f_best)

        metrics.append(metric)
        log_content += f"{i+1}. columns {list(cols)}: total rows = {metric}\n"

    if not metrics:
        raise ValueError("No overlapping best column combinations found between dev and new results.")

    avg_5 = sum(metrics) / len(metrics)
    avg_t_5=sum(count_t_bests) / len(count_t_bests)
    avg_f_5=sum(count_f_bests) / len(count_f_bests)
    return log_content, avg_5,avg_t_5,avg_f_5


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files based on provided paths.")

    parser.add_argument("--topic", type=str, default="", help="Topic name.")
    parser.add_argument("--log_file_input", type=str, default="/home/liujunfeng/sense-judge/benchmark_output_20250916_144813", help="Path to the new 'false' file.")
    parser.add_argument("--chazhi", type=str, default="0", help="Path to the new 'false' file.")
    parser.add_argument("--file_path_false", type=str, required=True, help="Dev/baseline reverse False result JSON.")
    parser.add_argument("--file_path_false_new", type=str, required=True, help="New/eval reverse False result JSON.")
    parser.add_argument(
        "--denominator",
        type=float,
        default=250,
        help="Deprecated; the reported percentage is computed from the actual paired denominator in the data.",
    )

    args = parser.parse_args()

    topic=args.topic
    file_path_false = args.file_path_false
    file_path_true=file_path_false.replace("False","True")

    file_path_false_new = args.file_path_false_new
    file_path_true_new = file_path_false_new.replace("False","True")


    log_file_input=args.log_file_input
    chazhi=args.chazhi
    matrix_true = load_and_process_data(file_path_true, True)
    matrix_false = load_and_process_data(file_path_false, False)
    if isinstance(matrix_true, str):
        raise SystemExit(matrix_true)
    if isinstance(matrix_false, str):
        raise SystemExit(matrix_false)
    
    # Analysis without difference penalty.
    sorted_results_no=analyze_combinations(topic,matrix_true, matrix_false, False)
    # Analysis with difference penalty.
    sorted_results_chazhi=analyze_combinations(topic,matrix_true, matrix_false, True)




    matrix_true = load_and_process_data(file_path_true_new, True)
    matrix_false = load_and_process_data(file_path_false_new, False)
    if isinstance(matrix_true, str):
        raise SystemExit(matrix_true)
    if isinstance(matrix_false, str):
        raise SystemExit(matrix_false)

    sorted_results_no_new=analyze_combinations(topic,matrix_true, matrix_false, False)
    # Analysis with difference penalty.
    sorted_results_chazhi_new=analyze_combinations(topic,matrix_true, matrix_false, True)
    content_need=""
    content_need+=f"-----------{file_path_false}---{topic}-------------\n"

    for cols, metric in sorted_results_no_new:
        if cols == (0,):
            count_t_best = count_rows_over_half(matrix_true, True, list(cols))
            count_f_best = count_rows_over_half(matrix_false, False, list(cols))
            # content_need+=f"Column [0] metric on new data: {metric}\n"
            # content_need+=f"Column [0] metric = {metric}\n"


    devset, strategy, forward_best, inverted_worst = select_combinations_by_dev_strategy(
        topic,
        load_and_process_data(file_path_true, True),
        load_and_process_data(file_path_false, False),
        chazhi == "1",
    )
    count_n = len(devset)
    print(
        f"strategy:{strategy},forward_best:{forward_best},"
        f"inverted_worst:{inverted_worst},topic:{topic},"
        f"preference_combination_num:{count_n}"
    )

    metrics = []
    denominators = []
    count_t_bests = []
    count_f_bests = []
    for cols, _ in devset:
        selected_cols = list(cols)
        forward_metric = count_paired_rows(matrix_true, matrix_false, selected_cols)
        denominator = count_paired_denominator(matrix_true, matrix_false, selected_cols)
        count_t_best = count_rows_over_half(matrix_true, True, selected_cols)
        count_f_best = count_rows_over_half(matrix_false, False, selected_cols)

        if strategy == "inverted_worst":
            metric = denominator - forward_metric
        else:
            metric = forward_metric

        metrics.append(metric)
        denominators.append(denominator)
        count_t_bests.append(count_t_best)
        count_f_bests.append(count_f_best)

    if not metrics:
        raise ValueError("No selected preference combinations found.")

    average_metric = sum(metrics) / len(metrics)
    average_denominator = sum(denominators) / len(denominators)
    sensejudge_score = average_metric * 100 / average_denominator if average_denominator else 0
    avg_t_5 = sum(count_t_bests) / len(count_t_bests)
    avg_f_5 = sum(count_f_bests) / len(count_f_bests)
    content_need+=f"sensejudge: {average_metric}\n"
    content_need+=f"actual_denominator: {average_denominator}\n"
    content_need+=f"sensejudge: {sensejudge_score}\n"
    print(
        f"strategy:{strategy},forward_best:{forward_best},"
        f"inverted_worst:{inverted_worst},topic:{topic},"
        f"preference_combination_num:{count_n},"
        f"actual_denominator:{average_denominator},"
        f"sensejudge: {sensejudge_score}\n"
    )
    
    # print(content_need)
