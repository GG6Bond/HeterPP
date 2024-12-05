source ./ip_list.sh


# # 本地目标目录
log_dir="/workspace/Hybrid_PTAFM_1F1B/logs/"
trace_dir="/workspace/Hybrid_PTAFM_1F1B/trace_json/"


# 遍历所有IP地址
for ip in "${ips[@]}"; do
    echo "正在从 $ip 复制文件..."
    
    # 使用scp命令从远程服务器复制文件到本地目录
    scp -P 222 -r root@"$ip":/workspace/Hybrid_PTAFM_1F1B/logs/* "$log_dir"
    scp -P 222 -r root@"$ip":/workspace/Hybrid_PTAFM_1F1B/trace_json/* "$trace_dir"
    
    # 检查scp命令是否成功
    if [ $? -eq 0 ]; then
        echo "从 $ip 复制成功"
    else
        echo "从 $ip 复制失败"
    fi
done

echo "所有文件复制完成"
