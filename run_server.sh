#!/bin/bash
# NICE-EEG 服务器运行脚本
# 使用方法: bash run_server.sh [参数]

# 默认参数
DNN="clip"
EPOCH=200
NUM_SUB=1
BATCH_SIZE=1000
SEED=2023
USE_SCREEN=true
USE_NOHUP=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dnn)
            DNN="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --num_sub)
            NUM_SUB="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --nohup)
            USE_NOHUP=true
            USE_SCREEN=false
            shift
            ;;
        --foreground)
            USE_SCREEN=false
            USE_NOHUP=false
            shift
            ;;
        -h|--help)
            echo "使用方法: bash run_server.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --dnn MODEL           DNN模型 (clip/vit/resnet) [默认: clip]"
            echo "  --epoch N             训练轮数 [默认: 200]"
            echo "  --num_sub N           被试数量 [默认: 1]"
            echo "  --batch-size N        批次大小 [默认: 1000]"
            echo "  --seed N              随机种子 [默认: 2023]"
            echo "  --nohup               使用nohup后台运行"
            echo "  --foreground          前台运行（不使用screen/nohup）"
            echo "  -h, --help            显示帮助信息"
            echo ""
            echo "示例:"
            echo "  bash run_server.sh --dnn clip --epoch 200 --num_sub 1"
            echo "  bash run_server.sh --nohup --batch-size 500"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未找到nvidia-smi，可能没有GPU"
else
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

# 检查必要文件
if [ ! -f "nice_stand.py" ]; then
    echo "错误: 未找到 nice_stand.py"
    exit 1
fi

if [ ! -d "dnn_feature" ]; then
    echo "警告: 未找到 dnn_feature 目录"
fi

# 构建命令
CMD="python nice_stand.py --dnn $DNN --epoch $EPOCH --num_sub $NUM_SUB --batch-size $BATCH_SIZE --seed $SEED"

echo "=========================================="
echo "准备运行训练"
echo "=========================================="
echo "命令: $CMD"
echo ""

# 根据运行模式执行
if [ "$USE_SCREEN" = true ]; then
    SESSION_NAME="nice-$(date +%Y%m%d-%H%M%S)"
    echo "使用screen运行，会话名: $SESSION_NAME"
    echo ""
    echo "提示:"
    echo "  - 分离会话: Ctrl+A, 然后按 D"
    echo "  - 重新连接: screen -r $SESSION_NAME"
    echo "  - 查看所有会话: screen -ls"
    echo ""
    read -p "按Enter开始运行..."
    screen -S "$SESSION_NAME" bash -c "$CMD; exec bash"
    
elif [ "$USE_NOHUP" = true ]; then
    LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
    echo "使用nohup后台运行，日志文件: $LOG_FILE"
    echo ""
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "进程ID: $PID"
    echo "查看日志: tail -f $LOG_FILE"
    echo "查看进程: ps aux | grep $PID"
    
else
    echo "前台运行..."
    echo ""
    $CMD
fi
