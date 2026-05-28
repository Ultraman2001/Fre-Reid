#!/bin/bash
# RATR Lambda 消融实验脚本 (GPU 0 版)
# 基于 Duke 数据集测试不同 RATR_LAMBDA 值
# 每次同时运行 2 个实验（全部在 GPU 0 上运行，请确保显存足够）

set -e

# 基础配置路径
BASE_CONFIG="configs/MSMT17/mambavision_tiny_transreid.yml"
OUTPUT_BASE="logs/MSMT17-RATR-2-drop0.5"

# 要测试的 RATR_LAMBDA 值 (移除了 0.0, 0.2; 增加了 1.5, 2.5)
LAMBDAS=(1.0 1.5 2.0 2.5)

# 创建输出目录
mkdir -p $OUTPUT_BASE

echo "=========================================="
echo "RATR Lambda 消融实验 (GPU 0)"
echo "基础配置: $BASE_CONFIG"
echo "测试值: ${LAMBDAS[@]}"
echo "注意: 2个实验将同时在 GPU 0 运行，请监控显存 (OOM 预警)"
echo "=========================================="

# 函数：运行单个实验
run_experiment() {
    local lambda=$1
    local output_dir="${OUTPUT_BASE}/lambda_${lambda}"
    
    echo "[GPU 0] Starting experiment: RATR_LAMBDA=$lambda"
    
    # 强制在卡 0 运行
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config_file $BASE_CONFIG \
        SOLVER.RATR_ENABLED True \
        SOLVER.RATR_LAMBDA $lambda \
        OUTPUT_DIR $output_dir \
        > "${output_dir}/train.log" 2>&1
    
    echo "[GPU 0] Completed: RATR_LAMBDA=$lambda"
}

# 两两并行运行
for ((i=0; i<${#LAMBDAS[@]}; i+=2)); do
    lambda1=${LAMBDAS[$i]}
    lambda2=${LAMBDAS[$((i+1))]:-""}
    
    echo ""
    echo "=========================================="
    echo "Round $((i/2+1)): Parallel testing lambda=$lambda1 and lambda=$lambda2 on GPU 0"
    echo "=========================================="
    
    # 创建输出目录并启动第一个
    mkdir -p "${OUTPUT_BASE}/lambda_${lambda1}"
    run_experiment $lambda1 &
    PID1=$!
    
    # 如果有第二个 lambda 值，启动第二个
    if [ -n "$lambda2" ]; then
        mkdir -p "${OUTPUT_BASE}/lambda_${lambda2}"
        run_experiment $lambda2 &
        PID2=$!
        
        # 等待两个实验完成
        wait $PID1
        wait $PID2
    else
        # 只有一个实验，等待完成
        wait $PID1
    fi
    
    echo "Round $((i/2+1)) completed!"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="

# 汇总结果
echo ""
echo "===== 结果汇总 ====="
for lambda in "${LAMBDAS[@]}"; do
    log_file="${OUTPUT_BASE}/lambda_${lambda}/train.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "--- RATR_LAMBDA=$lambda ---"
        # 提取最后一次评估结果
        grep -E "(mAP|Rank-1)" "$log_file" | tail -4
    fi
done
