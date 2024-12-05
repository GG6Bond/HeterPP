import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('batch_size',type=int,help='')
parser.add_argument('path',type=str,help='')
args = parser.parse_args()

batch_size = args.batch_size
path = args.path

num = batch_size - 1

# 读取 JSON 文件
with open(path, 'r') as file:
    data = json.load(file)

# 提取每个对象中的 tid、dur 和 micro-batch
extracted_data = []

# micro-batch 时间
forward_recv_time = 0
forward_compute_time = 0
forward_send_time = 0
backward_recv_time = 0
backward_compute_time = 0
backward_send_time = 0
optimizer_comm_time = 0
optimizer_comp_time = 0
optimizer_step_time = 0

# mini-batch时间
# forward_recv_time_start = 0
# forward_recv_time_end = 0
# mini_batch_forward_recv_time = []

# forward_compute_time_start = 0
# forward_compute_time_end = 0
# mini_batch_forward_compute_time = []

# forward_send_time_start = 0
# forward_send_time_end = 0
# mini_batch_forward_send_time = []

# backward_recv_time_start = 0
# backward_recv_time_end = 0
# mini_batch_backward_recv_time = []

# backward_compute_time_start = 0
# backward_compute_time_end = 0
# mini_batch_backward_compute_time = []

# backward_send_time_start = 0
# backward_send_time_end = 0
# mini_batch_backward_send_time = []



for item in data:
    tid = item.get('tid')
    ts = item.get('ts')
    dur = item.get('dur')
    micro_batch = item.get('args', {}).get('micro-batch')
    

    if tid == '1. forward-recv':
        if micro_batch == 0:
            continue
        forward_recv_time += dur
        
    if tid == '2. forward-compute':
        if micro_batch == 0:
            continue
        forward_compute_time += dur
    
    if tid == '3. forward-send':
        if micro_batch == 0:
            continue
        forward_send_time += dur


    if tid == '4. backward-recv':
        if micro_batch == 0:
            continue
        backward_recv_time += dur

    if tid == '5. backward-compute':
        if micro_batch == 0:
            continue
        backward_compute_time += dur

    if tid == '6. backward-send':
        if micro_batch == 0:
            continue
        backward_send_time += dur


    if tid == '7. optimizer-comm':
        optimizer_comm_time += dur

    if tid == '8. optimizer-comp':
        optimizer_comp_time += dur

    if tid == '7. optimizer-step':
        optimizer_step_time += dur


print('====== forward ======')
print('forward_recv_time:',forward_recv_time//num)
print('forward-compute_time:',forward_compute_time//num)
print('forward_send_time:',forward_send_time//num)

print('====== backward ======')
print('backward_recv_time:',backward_recv_time//num)
print('backward_compute_time:',backward_compute_time//num)
print('backward_send_time:',backward_send_time//num)

print('====== optimizer ======')
print('optimizer_comm_time:',optimizer_comm_time//5)
print('optimizer_comp_time:',optimizer_comp_time//5)
print('optimizer_step_time:',optimizer_step_time//5)