TEST_DATA="data/cve_fixes_and_big_vul/test.jsonl"
GPU_ID="0"
CKPT_PATH="checkpoint/best_model.pt"
LOG_PREFIX="test"

python test_model.py \
    --test_data_path="$TEST_DATA" \
    --model_path="/mnt/e/LLM/Salesforce/codet5-base" \
    --ckpt_path="$CKPT_PATH" \
    --ablation="full" \
    --gpu_id="$GPU_ID" \
    --num_beams 1 2>&1 | tee "log/${LOG_PREFIX}_beam1.log"

python test_model.py \
    --test_data_path=$TEST_DATA \
    --model_path="/mnt/e/LLM/Salesforce/codet5-base" \
    --ckpt_path=$CKPT_PATH \
    --ablation="full" \
    --gpu_id="$GPU_ID" \
    --num_beams 3 2>&1 | tee "log/${LOG_PREFIX}_beam3.log"

python test_model.py \
    --test_data_path=$TEST_DATA \
    --model_path="/mnt/e/LLM/Salesforce/codet5-base" \
    --ckpt_path=$CKPT_PATH \
    --ablation="full" \
    --gpu_id="$GPU_ID" \
    --num_beams 5 2>&1 | tee "log/${LOG_PREFIX}_beam5.log"
