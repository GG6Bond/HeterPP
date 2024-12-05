source ./ip_list.sh

for ip in "${ips[@]}"
do
    echo "Send "$ip" files ================================================================================================================================="
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/scripts root@"$ip":/workspace/Hybrid_PTAFM_1F1B
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/pipeline_parallel root@"$ip":/workspace/Hybrid_PTAFM_1F1B
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/comm root@"$ip":/workspace/Hybrid_PTAFM_1F1B
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/utils root@"$ip":/workspace/Hybrid_PTAFM_1F1B
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/foo.py root@"$ip":/workspace/Hybrid_PTAFM_1F1B
    scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B/dist_runner.py root@"$ip":/workspace/Hybrid_PTAFM_1F1B

    # scp -r -P 222 /workspace/Hybrid_PTAFM_1F1B root@"$ip":/workspace
      if [ $? -eq 0 ]; then
        echo "File sent to $ip successfully."
    else
        echo "Failed to send file to $ip."
    fi
done