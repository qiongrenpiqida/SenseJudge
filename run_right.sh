#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NAME="${NAME:-sense-label3}"
INPUT_JSON_FILE="${INPUT_JSON_FILE:-data/input/human_label/${NAME}.json}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-pro}"
PREFERENCE_MODEL_NAME="${PREFERENCE_MODEL_NAME:-${MODEL_NAME}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
DENOMINATOR="${DENOMINATOR:-250}"
RUN_PREFERENCE="${RUN_PREFERENCE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
TOPICS="${TOPICS:-Math}"

DEV_JSON_FILE="data/data_process/${NAME}-dev.json"
TEST_JSON_FILE="data/data_process/${NAME}-test.json"
PREFERENCE_DIR="data/preference/${NAME}-dev-preference"
DEV_RESULT_DIR="data/result/${NAME}-dev-${MODEL_NAME}"
TEST_RESULT_DIR="data/result/${NAME}-test-${MODEL_NAME}"

read -r -a topics <<< "${TOPICS}"

if [[ "${MODEL_NAME}" == deepseek* || "${PREFERENCE_MODEL_NAME}" == deepseek* ]]; then
    if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${MODEL_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
        echo "DeepSeek models require DEEPSEEK_API_KEY, MODEL_API_KEY, or OPENAI_API_KEY." >&2
        exit 1
    fi
fi

echo "Model API base: ${MODEL_API_BASE:-${OPENAI_API_BASE:-${SELF_HOSTED_API_BASE:-${DEEPSEEK_API_BASE:-${LOCAL_MODEL_API_BASE:-auto}}}}}"

echo "Step 0/4: preparing dev/test split from ${INPUT_JSON_FILE}"
"${PYTHON_BIN}" 0_get_dev_for_preference.py --input_json_file "${INPUT_JSON_FILE}"

if [[ "${RUN_PREFERENCE}" == "1" ]]; then
    echo "Step 1/4: generating preferences with ${PREFERENCE_MODEL_NAME}"
    for topic in "${topics[@]}"; do
        echo "Generating preferences for topic: ${topic}"
        "${PYTHON_BIN}" 1_get_preference.py \
            --dev_json_file "${DEV_JSON_FILE}" \
            --topic "${topic}" \
            --model_name "${PREFERENCE_MODEL_NAME}"
    done
else
    echo "Step 1/4: reusing existing preferences from ${PREFERENCE_DIR}"
fi

echo "Step 2/4: running dev baseline judge with ${MODEL_NAME}"
for topic in "${topics[@]}"; do
    if [[ "${FORCE_RERUN}" == "1" ]]; then
        rm -f "${DEV_RESULT_DIR}/${topic}_output_reverse_False.json" \
              "${DEV_RESULT_DIR}/${topic}_output_reverse_True.json"
    fi
    echo "Running dev judge for topic: ${topic}"
    "${PYTHON_BIN}" 2_judge.py \
        --model_name "${MODEL_NAME}" \
        --data_path "${DEV_JSON_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --preference_path "${PREFERENCE_DIR}/${topic}.json" \
        --topic "${topic}" \
        --file_path_false test_mode \
        --max_samples "${MAX_SAMPLES}"
done

echo "Step 3/4: running test judge using dev-selected preferences"
for topic in "${topics[@]}"; do
    if [[ "${FORCE_RERUN}" == "1" ]]; then
        rm -f "${TEST_RESULT_DIR}/${topic}_output_reverse_False.json" \
              "${TEST_RESULT_DIR}/${topic}_output_reverse_True.json"
    fi
    echo "Running test judge for topic: ${topic}"
    "${PYTHON_BIN}" 2_judge.py \
        --model_name "${MODEL_NAME}" \
        --data_path "${TEST_JSON_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --preference_path "${PREFERENCE_DIR}/${topic}.json" \
        --topic "${topic}" \
        --file_path_false "${DEV_RESULT_DIR}/${topic}_output_reverse_False.json" \
        --max_samples "${MAX_SAMPLES}"
done

echo "Step 4/4: counting SenseJudge results"
for topic in "${topics[@]}"; do
    echo "Counting topic: ${topic}"
    "${PYTHON_BIN}" 3_judge_count.py \
        --file_path_false "${DEV_RESULT_DIR}/${topic}_output_reverse_False.json" \
        --file_path_false_new "${TEST_RESULT_DIR}/${topic}_output_reverse_False.json" \
        --topic "${topic}" \
        --chazhi "0" \
        --denominator "${DENOMINATOR}"
done

echo "Pipeline finished."
echo "Dev result dir: ${DEV_RESULT_DIR}"
echo "Test result dir: ${TEST_RESULT_DIR}"
