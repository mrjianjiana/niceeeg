#!/bin/bash
# NICE-EEG 服务器快速配置脚本
# 使用方法: bash setup_server.sh

echo "=========================================="
echo "NICE-EEG 服务器配置脚本"
echo "=========================================="

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 配置变量（请根据实际情况修改）
read -p "请输入项目根目录路径 [默认: $SCRIPT_DIR]: " PROJECT_DIR
PROJECT_DIR=${PROJECT_DIR:-$SCRIPT_DIR}

read -p "请输入数据目录路径 [默认: /home/\$USER/Data/Things-EEG2]: " DATA_DIR
DATA_DIR=${DATA_DIR:-"/home/$USER/Data/Things-EEG2"}

read -p "请输入结果保存路径 [默认: /home/\$USER/results/NICE]: " RESULTS_DIR
RESULTS_DIR=${RESULTS_DIR:-"/home/$USER/results/NICE"}

read -p "请输入GPU编号 [默认: 0]: " GPU_ID
GPU_ID=${GPU_ID:-0}

echo ""
echo "配置信息："
echo "  项目目录: $PROJECT_DIR"
echo "  数据目录: $DATA_DIR"
echo "  结果目录: $RESULTS_DIR"
echo "  GPU编号: $GPU_ID"
echo ""

read -p "确认配置是否正确? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "配置已取消"
    exit 1
fi

# 备份原文件
echo "备份原文件..."
cp nice_stand.py nice_stand.py.bak

# 修改 nice_stand.py
echo "修改 nice_stand.py..."

# 修改GPU设置
sed -i "s|gpus = \[6\]|gpus = [$GPU_ID]|g" nice_stand.py

# 修改结果路径
sed -i "s|result_path = '/home/NICE/results/'|result_path = '$RESULTS_DIR/'|g" nice_stand.py

# 修改EEG数据路径
sed -i "s|self.eeg_data_path = '/home/Data/Things-EEG2/Preprocessed_data_250Hz/'|self.eeg_data_path = '$DATA_DIR/Preprocessed_data_250Hz/'|g" nice_stand.py

# 创建必要目录
echo "创建必要目录..."
mkdir -p model
mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$DATA_DIR")"

echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 确保数据已上传到: $DATA_DIR"
echo "2. 检查GPU: nvidia-smi"
echo "3. 运行训练: python nice_stand.py --dnn clip --epoch 200 --num_sub 1"
echo ""
echo "备份文件: nice_stand.py.bak"
echo ""
