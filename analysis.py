import json
import re
from itertools import combinations


FINAL_DECISION_PATTERNS = [
    re.compile(r"The\s*final\s*decision\s*is\s*(?:\*\*)?\s*(?:Response\s*)?([AB])", re.I),
    re.compile(r"final\s*decision\s*[:：]?\s*(?:\*\*)?\s*(?:Response\s*)?([AB])", re.I),
    # Keep this Chinese-compatible parser because some judge models may answer in Chinese.
    re.compile(r"最终(?:决定|决策|选择|答案|回答)(?:是|为|：|:)?\s*(?:\*\*)?\s*(?:回应|回复|响应|回答|Response)?\s*([ABＡＢ])", re.I),
]


def extract_final_decision(text):
    hits = []
    text = text or ""
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
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Error: file not found: {file_path}\n"
    except json.JSONDecodeError:
        return f"Error: failed to decode JSON file: {file_path}\n"

    output_matrix = []
    for item in data:
        if item and "outputs" in item and isinstance(item["outputs"], list):
            row = []
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
            output_matrix.append(row)
        else:
            output_matrix.append([])
    return output_matrix


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
        # Tie means defaulting to Response A.
        is_correct = not reverse
    return is_correct, is_tie, True


def count_rows_over_half(matrix, reverse, selected_cols):
    rows_over_half = 0
    for row in matrix:
        is_correct, _, has_valid = row_vote(row, reverse, selected_cols)
        if has_valid and is_correct:
            rows_over_half += 1
    return rows_over_half


def count_paired_rows(matrix_true, matrix_false, selected_cols):
    rows_over_half = 0
    max_rows = max(len(matrix_true), len(matrix_false))
    for row_index in range(max_rows):
        if row_index < len(matrix_true):
            true_correct, _, _ = row_vote(matrix_true[row_index], True, selected_cols)
        else:
            true_correct = False

        if row_index < len(matrix_false):
            false_correct, _, _ = row_vote(matrix_false[row_index], False, selected_cols)
        else:
            false_correct = False

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


def analyze_combinations(topic, matrix_true, matrix_false, consider_difference):
    if isinstance(matrix_true, str) or isinstance(matrix_false, str):
        return matrix_true if isinstance(matrix_true, str) else matrix_false

    max_cols_true = max((len(row) for row in matrix_true), default=0)
    max_cols_false = max((len(row) for row in matrix_false), default=0)
    max_cols = max(max_cols_true, max_cols_false)
    results = {}

    for size in range(1, max_cols + 1):
        for selected_cols_tuple in combinations(range(max_cols), size):
            selected_cols = sorted(list(selected_cols_tuple))
            count_true = count_rows_over_half(matrix_true, True, selected_cols)
            count_false = count_rows_over_half(matrix_false, False, selected_cols)
            paired_count = count_paired_rows(matrix_true, matrix_false, selected_cols)

            if consider_difference:
                evaluation_metric = paired_count - abs(count_true - count_false)
            else:
                evaluation_metric = paired_count
            results[tuple(selected_cols)] = evaluation_metric

    return sorted(results.items(), key=lambda item: item[1], reverse=True)


def select_combinations_by_dev_strategy(topic, matrix_true, matrix_false, consider_difference=False):
    if isinstance(matrix_true, str) or isinstance(matrix_false, str):
        return matrix_true if isinstance(matrix_true, str) else matrix_false

    max_cols_true = max((len(row) for row in matrix_true), default=0)
    max_cols_false = max((len(row) for row in matrix_false), default=0)
    max_cols = max(max_cols_true, max_cols_false)
    scored_results = []

    for size in range(1, max_cols + 1):
        for selected_cols_tuple in combinations(range(max_cols), size):
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
