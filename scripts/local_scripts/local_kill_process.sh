# pkill -9 python
pkill -f "python dist_runner.py"
# 定义端口号
#port=9000

# 使用 lsof 检查端口
#pid=$(lsof -ti tcp:$port)

# 检查变量是否为空
#if [ -z "$pid" ]; then
#    echo "No process found using port $port"
#else
#    # 终止找到的进程
#    kill -9 $pid
#    echo "Process using port $port has been terminated"
#fi
