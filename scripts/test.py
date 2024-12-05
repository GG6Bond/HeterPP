

# import torch


# # t = [torch.ones((3, 4, 5),requires_grad=True,device='cuda:1') for _ in range(16)]
# # print(t)

# # print(type(t))
# # print('11212')
# # print(t[0])

# # s = torch.ones(1,2,3)
# # print(s)

# # ss = s.repeat(2,2,2)
# # print(ss)


# # t = [None] * 16
# # t[0] = [torch.ones((3, 4, 5),requires_grad=True,device='cuda:1') for _ in range(16)]
# # t[2] = [torch.ones((10, 4, 5),requires_grad=True,device='cuda:1') for _ in range(16)]

# # # print(type(t))
# # print(t)


# mapping =  [0, 0, 0, 1, 2, 3, 3, 3, 4]

# pp_rank = 8

# mapping_rank = mapping[pp_rank]

# print(mapping_rank)


import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('batch_size',type=int,help='')
parser.add_argument('path',type=str,help='')
args = parser.parse_args()

batch_size = args.batch_size
path = args.path

print('****** batch_size = ',batch_size,'file_path : ',path)
print()


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


# num = (batch_size - 1) * 2 * 5

for item in data:
    tid = item.get('tid')
    ts = item.get('ts')
    dur = item.get('dur')
    micro_batch = item.get('args', {}).get('micro-batch')
    

    # if tid == '1. forward-recv':
    #     if micro_batch == 0:
    #         continue
    #     if micro_batch == 1:
    #         forward_recv_time_start = ts
    #     if micro_batch == batch_size - 1:
    #         forward_recv_time_end = ts
    #         mini_batch_forward_recv_time.append(forward_recv_time_end - forward_compute_time_start)
        
            
        
    #     forward_recv_time += dur
    #     # print(dur)

    if tid == '2. forward-compute':
        if micro_batch == 0:
            continue
        forward_compute_time += dur
            
        
        
        forward_compute_time += dur
        # print(dur)
    
    # if tid == '3. forward-send':
    #     if micro_batch == 0:
    #         continue
    #     if micro_batch == 1:
    #         forward_send_time_start = ts
    #     if micro_batch == batch_size - 1:
    #         forward_send_time_end = ts
    #         mini_batch_forward_send_time.append(forward_send_time_end - forward_send_time_start)
    #     forward_send_time += dur
    #     # print(dur)

    # if tid == '4. backward-recv':
    #     if micro_batch == 0:
    #         continue
    #     if micro_batch == 1:
    #         backward_recv_time_start = ts
    #     if micro_batch == batch_size - 1:
    #         backward_recv_time_end = ts
    #         mini_batch_backward_recv_time.append(backward_recv_time_end - backward_recv_time_start)
    #     backward_recv_time += dur
    #     # print(dur)


    if tid == '5. backward-compute':
        if micro_batch == 0:
            continue
        backward_compute_time += dur
        # if micro_batch == 1:
        #     backward_compute_time_start = ts
        # if micro_batch == batch_size - 1:
        #     backward_compute_time_end = ts
        #     mini_batch_backward_compute_time.append(backward_compute_time_end - backward_compute_time_start)
        # print(dur)

    # if tid == '6. backward-send':
    #     if micro_batch == 0:
    #         continue
        
    #     if micro_batch == 1:
    #         backward_send_time_start = ts
    #     if micro_batch == batch_size - 1:
    #         backward_send_time_end = ts
    #         mini_batch_backward_send_time.append(backward_send_time_end - backward_send_time_start)
        
    #     forward_compute_time += dur
    #     # print(dur)

    # if tid == '7. optimizer-comm':
    #     if micro_batch == 0:
    #         continue
    #     optimizer_comm_time += dur
    #     # print(dur)

    # if tid == '8. optimizer-comp':
    #     if micro_batch == 0:
    #         continue
    #     optimizer_comp_time += dur
        # print(dur)

    # extracted_data.append({
    #     'tid': tid,
    #     'dur': dur,
    #     'micro-batch': micro_batch
    # })
    
# print('total_forward_recv_time:',forward_recv_time)
# print('avg_microbatch_forward_recv_time:',forward_recv_time//num)

# print('total_forward_compute_time:',forward_compute_time)
# print('avg_microbatch_forward_compute_time:',forward_compute_time//num)

# print('total_forward_send_time:',forward_send_time//num)
# print('avg_microbatch_forward_send_time:',forward_send_time//num)

# print('total_backward_recv_time:',backward_recv_time//num)
# print('avg_microbatch_backward_recv_time:',backward_recv_time//num)


# print('total_backward_compute_time:',backward_compute_time//num)
# print('avg_microbatch_backward_compute_time:',backward_compute_time//num)

# print('total_backward_send_time:',backward_send_time//num)
# print('avg_microbatch_backward_send_time:',backward_send_time//num)



# print('optimizer_comm_time:',optimizer_comm_time//5)
# print('optimizer_comp_time:',optimizer_comp_time//5)




# print('mini_batch_forward_recv_time:        ',mini_batch_forward_recv_time)
# print('mini_batch_forward_compute_time:     ',mini_batch_forward_compute_time)
# print('mini_batch_forward_send_time:        ',mini_batch_forward_send_time)
# print('mini_batch_backward_recv_time:       ',mini_batch_backward_recv_time)
# print('mini_batch_backward_compute_time:    ',mini_batch_backward_compute_time)
# print('mini_batch_backward_send_time:       ',mini_batch_backward_send_time)
# print()
# mini_batch_forward_recv_time = mini_batch_forward_recv_time[2:10]
# # print(mini_batch_forward_recv_time)
# if len(mini_batch_forward_recv_time) != 0:
#     print('avg_mini_batch_forward_recv_time:     ',sum(mini_batch_forward_recv_time)//len(mini_batch_forward_recv_time))

# mini_batch_forward_compute_time = mini_batch_forward_compute_time[2:10]
# # print(mini_batch_forward_compute_time)
# print('avg_mini_batch_forward_compute_time:     ',sum(mini_batch_forward_compute_time)//len(mini_batch_forward_compute_time))

# mini_batch_forward_send_time = mini_batch_forward_send_time[2:10]
# # print(mini_batch_forward_send_time)
# if len(mini_batch_forward_send_time):
#     print('avg_mini_batch_forward_send_time:     ',sum(mini_batch_forward_send_time)//len(mini_batch_forward_send_time))


# mini_batch_backward_recv_time = mini_batch_backward_recv_time[2:10]
# # print(mini_batch_backward_recv_time)
# print('avg_mini_batch_backward_recv_time:     ',sum(mini_batch_backward_recv_time)//len(mini_batch_backward_recv_time))

# mini_batch_backward_compute_time = mini_batch_backward_compute_time[2:10]
# # print(mini_batch_backward_compute_time)
# print('avg_mini_batch_backward_compute_time:     ',sum(mini_batch_backward_compute_time)//len(mini_batch_backward_compute_time))


# mini_batch_backward_send_time = mini_batch_backward_send_time[2:10]
# # print(mini_batch_backward_send_time)
# if len(mini_batch_backward_send_time):
#     print('avg_mini_batch_backward_send_time:     ',sum(mini_batch_backward_send_time)//len(mini_batch_backward_send_time))



# print('avg_optimizer_comm_time:',optimizer_comm_time//5)
# print('avg_optimizer_comp_time:',optimizer_comp_time//5)
# print()

print(forward_compute_time)
print(backward_compute_time)






mapping =  [0, 0, 0, 1, 2, 3, 3, 3, 4]
mapping_rank = mapping[self.pp_rank]

print(f"pp_rank: {self.pp_rank}, mapping_rank: {mapping_rank}, forward compute micro-batches: {4 - mapping_rank}")


# Starting phase: to fill the pipeline_parallel.
while forward_i < 4 - mapping_rank:
    self.forward_micro_batch(forward_index=forward_i)
    forward_i += 1

# Running phase: 1 forward coupled with 1 backward.
while forward_i < self.micro_batch_num:
    self.forward_micro_batch(forward_index=forward_i)
    self.backward_micro_batch(backward_index=backward_i,
                                cached_output_micro_batches=self.current_micro_output,
                                target_as_micro_batches=target_as_micro_batches)
    forward_i += 1
    backward_i += 1


# Ending phase: to finish the rest stages in the pipeline_parallel.
while backward_i < self.micro_batch_num:
    self.backward_micro_batch(backward_index=backward_i,
                                cached_output_micro_batches=self.current_micro_output,
                                target_as_micro_batches=target_as_micro_batches)
    backward_i += 1