# 1️⃣ 指定 data-juicer 代码的正确路径
DJ_PATH="/root/autodl-tmp/mini-deepseekv3/data/step2/data-juicer"

# 2️⃣ 依次执行所有 5 个 YAML 配置文件
CONFIG_FILES=(
    #"minidsv3_text_openr1.yaml"
    #"minidsv3_text_ape.yaml"
    "minidsv3_code_starcoder.yaml"
    #"minidsv3_text_skypile.yaml"
    #"minidsv3_text_slimpajama.yaml"
)

# 3️⃣ 逐个运行 process_data.py，并使用 tee 实时查看输出
for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Processing $CONFIG..."
    nohup python $DJ_PATH/tools/process_data.py --config "$CONFIG" > >(tee "dj_log_${CONFIG%.*}.log") 2>&1 &
    sleep 5  # ✅ 防止进程同时启动太多，增加稳定性
done

# 等待所有后台进程完成
wait

echo "✅ All processes completed! Check logs: dj_log_*"

