import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer


class GpipeAsync:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device, use_dp=False):
        print("=======Initialize Gpipe.")
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32")
        self.use_dp = use_dp
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.global_rank = args.rank
        self.pipeline_group_size = get_pipeline_parallel_world_size()
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1  # 第一个   的 前一个是-1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1  # 最后一个 的 后一个是-1
        self.comm_size = get_pipeline_parallel_world_size()
        self.comm = get_pipeline_parallel_comm()
        # 获取gather结果
        self.gather_comm = get_pipeline_gather_comm()

        self.scatter_comm = get_pipeline_scatter_comm()

        self.device_gpu = get_pipeline_scatter_comm()

        self.device_gpu = get_device_gpu()
        self.first_node = False

        print(
            f"global_rank: {self.global_rank}, pipeline_group_size: {self.pipeline_group_size}, pp_rank: {self.pp_rank}, pre_node_rank: {self.pre_node_rank}, post_node_rank: {self.post_node_rank}, comm_size: {self.comm_size}, comm: {self.comm}, gather_comm: {self.gather_comm}, scatter_comm: {self.scatter_comm}, device_gpu: {self.device_gpu}")


        self.gradient_accumulate_step = args.gradient_accumulate_step
        print("=======Gradient accumulate step: ", self.gradient_accumulate_step)

        assert (args.batch_size % args.micro_batch_size == 0)
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        #self.enable_tidy_profiling=False
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]

        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                            for _ in range(self.micro_batch_num)]

            self.backward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                             for _ in range(self.micro_batch_num)]
            self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.init_time_stamp = None
            self.optimizer_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_end_event = torch.cuda.Event(enable_timing=True, blocking=False)

        self._compute_micro_batch_size()


        if self.device_gpu == 1:
            if self.pp_rank == 0:
                self.first_node = True
        else:
            if self.pp_rank <= 2:
                self.first_node = True


        if self.first_node:
            self.input_micro_batches = None
            self.concatenated_tensor = None
        else:
            if self.device_gpu == 0:
                self.input_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device, dtype=self.dtype)
                                        for _ in range(self.micro_batch_num)]
            else:
                self.input_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device, dtype=self.dtype)
                                        for _ in range(self.micro_batch_num)]
                self.concatenated_tensor = [torch.zeros((self.micro_batch_size * 3, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device, dtype=self.dtype)
                                        for _ in range(self.micro_batch_num)]

        if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            self.output_micro_batches_grad = None
        else:
            if self.device_gpu==0:
                self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                          requires_grad=False, device=self.device, dtype=self.dtype)
                                              for _ in range(self.micro_batch_num)]
            else:
                self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                          requires_grad=False, device=self.device, dtype=self.dtype)
                                              for _ in range(self.micro_batch_num)]
                self.concat_micro_batches_grad = [
                    torch.zeros((self.micro_batch_size * 3, self.seq_length, self.embedding_dim),
                            requires_grad=False, device=self.device, dtype=self.dtype)
                    for _ in range(self.micro_batch_num)]


        if self.first_node == True:
            self.model = GPTStageFirst(args, vocab_size, num_classes, device)
            print("self.globalID: "+str(self.global_rank)+".  completed First model load")
        elif self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            self.model = GPTStageLast(args, vocab_size, num_classes, device)
            print("self.globalID: " + str(self.global_rank) + ".  completed Last model load")
        else:
            self.model = GPTStageMiddle(args, vocab_size, num_classes, device)
            print("self.globalID: " + str(self.global_rank) + ".  completed Middle model load")


        if self.use_fp16:
            self.model.half()
            tmp_optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            self.optimizer = get_fp16_optimizer(args, tmp_optimizer, device)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        if use_dp:
            self.dp_optim = get_dp_module(args, device, self.model, self.optimizer)

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))
        else:
            print("=======Current micro-batch send/recv size: {} MB (fp32)"
                  .format(micro_batch_float_num * 4 // 1024 // 1024))
        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def zero_input_grad(self):
        if self.device_gpu ==0:        
            if self.input_micro_batches:
                for input_micro_batch in self.input_micro_batches:
                    if input_micro_batch.grad is not None:
                        input_micro_batch.grad.zero_()
        else:
            if self.concatenated_tensor:
                for input_micro_tensor in self.concatenated_tensor:
                    if input_micro_tensor.grad is not None:
                        input_micro_tensor.grad.zero_()

    def profile_mark_forward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_comp_start_events[i])

    def profile_mark_forward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_recv_start_events[i])

    def profile_mark_forward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_start_events[i])

    def profile_mark_forward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_end_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.backward_comp_start_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_start_events[i])

    def profile_mark_backward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_end_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_stage(self, input_data=None, target_data=None):
        # print("Forward stage start! rank-", self.rank)
        
        if self.first_node:
            assert (input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)

        elif self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            if self.model.task == 'Seq2SeqClassification':
                assert target_data is not None
                target_data_micro_batches = torch.chunk(target_data, self.micro_batch_num, dim=0)
        
        
        output_micro_batches = []
        
        self.gather_recv = False
        self.gather_send = False
        self.scatter_send = False
        self.scatter_recv = False
        if self.gather_comm is not None:
            self.pp_rank_gather = get_pipeline_gather_rank()
            self.gather_group_size = get_gather_world_size()
            if self.pp_rank_gather == self.gather_group_size - 1:
                self.gather_recv = True
            else:
                self.gather_send = True
        if self.scatter_comm is not None:
            self.pp_rank_scatter = get_pipeline_scatter_rank()
            self.scatter_group_size = get_scatter_world_size()
            if self.pp_rank_scatter == 0:
                self.scatter_send = True
            else:
                self.scatter_recv = True


        for i in range(self.micro_batch_num):
            gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in range(4)]

            if self.first_node:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    # 如果是A100要复制3份数据，
                    if self.device_gpu==1:
                        self.concatenated_tensor[i] = self.input_micro_batches[i].repeat(3)
                        current_micro_output = self.model(self.concatenated_tensor[i])
                    else:
                        current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    if self.gather_comm is not None:

                        self.gather_group_size = get_gather_world_size()
                        self.gather_comm.gather(current_micro_output.data, gather_list=gather_data,
                                                dst=self.gather_group_size - 1, stream=cupy_send_stream)
                    else:
                        self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)


            elif self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    # 最后的这个接收 实际上只有 scatter
                    # 验证逻辑：1、是否为gather通信组
                    #             如果是，进一步验证是否为 最后的汇聚节点，如果是执行，否则跳出本判断
                    #             如果不是，判断是否为A100节点，使用A100的接收缓存区。大小为6*self.input_micro_batches
                    #             最后则为T4节点
                    #         2、是否为Scatter通讯组
                    #             如果是，进一步验证是否为非0几点（因为0节点是发送节点），否则跳出
                    #             如果不是判断是否为A100
                    #             最后则为普通T4节点
                    if self.gather_recv:
                        self.gather_comm.gather(self.input_micro_batches[i], gather_list=gather_data,
                                                dst=self.gather_group_size - 1, stream=cupy_recv_stream)
                        # 方案1 将list数组使用torch的concat进行合并
                        gather_data.pop(self.pp_rank_gather)
                        self.concatenated_tensor[i] = torch.cat(gather_data, dim=0)
                        self.concatenated_tensor[i].requires_grad_(True)
                    elif self.scatter_recv:
                        #gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in range(4)]
                        self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=gather_data, src=0,
                                                  stream=cupy_recv_stream)
                    elif self.device_gpu == 1:
                        #  不属于gather和Scatter小圈子，说明上级是A100或是T4
                        self.comm.recv(self.concatenated_tensor[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    else:
                        # 否则为T4节点
                        if self.pipeline_group_size > 10:
                            src = self.pre_node_rank - 2
                            self.comm.recv(self.input_micro_batches[i], src=src, stream=cupy_recv_stream)
                        else:
                            self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)

        #                self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    #  如果Scatter不为空，判断是否为非0节点
                    # if self.scatter_comm is not None :
                    #     self.scatter_group_size = get_scatter_world_size()
                    #     if self.pp_rank !=0:
                    #         self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=gather_data, src=0,
                    #                                   stream=cupy_recv_stream)
                    # elif self.device_gpu =="A100":
                    #     #  不属于gather小圈子，说明上级是A100或是T4
                    #     self.comm.recv(self.concatenated_tensor[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    # else:
                    #     # 否则为T4节点
                    #     self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)

                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    if self.model.task == 'Seq2SeqClassification':
                        print("target_data_micro_batches[i]" + str(target_data_micro_batches[i].shape))
                        current_micro_output = self.model(self.input_micro_batches[i], target_data_micro_batches[i])
                    else:
                        #print(self.model.task)
                        # current_micro_output = self.model(self.input_micro_batches[i])
                        # 计算相对更简单，直接判断是否为A100，选择执行的数据集
                        if self.device_gpu == 1:
                            current_micro_output = self.model(self.concatenated_tensor[i])
                        else:
                            current_micro_output = self.model(self.input_micro_batches[i])

                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    # gather接收节点，只有最后的目的节点接收，其他的都是发送节点
                    if self.gather_recv:
                        #gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                         #              range(self.gather_group_size)]
                        self.gather_comm.gather(self.input_micro_batches[i], gather_list=gather_data,
                                                dst=self.gather_group_size - 1, stream=cupy_recv_stream)
                        # gather中的A100节点需要对聚合数据进行处理
                        gather_data.pop(self.pp_rank_gather)
                        self.concatenated_tensor[i] = torch.cat(gather_data, dim=0)
                        self.concatenated_tensor[i].requires_grad_(True)
                        # print("!!! recv sucess!  gatherID:"+str(self.pp_rank_gather)+". pp_rank: "+str(self.pp_rank))
                        # print()
                    elif self.scatter_recv:
                        #print("!!! Scatter start recv !  ScatterID:" + str(self.pp_rank_scatter) + ". pp_rank: " + str(
                        #    self.pp_rank) + ". global_rank: " + str(self.global_rank))
                        #gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in range(4)]
                        self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=gather_data, src=0,
                                                  stream=cupy_recv_stream)
                        #print("!!! Scattercomplete recv !  ScatterID:"+str(self.pp_rank_scatter)+". pp_rank: "+str(self.pp_rank))
                    #                        assert args.world_size == args.data_group_size * args.pipeline_group_size
                    elif self.device_gpu == 1:
                        #  不属于gather小圈子，说明上级是A100或是T4
                     #   print("global——rankID："+str(self.global_rank))
                        self.comm.recv(self.concatenated_tensor[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                        # print(self.global_rank.shape)
                    #    print("global——rankID  completed recv data flow ：" + str(self.global_rank))
                    else:
                        # 否则为T4节点
                        if self.pipeline_group_size > 10:
                            src = self.pre_node_rank - 2
                  #          print("ID number :" + str(self.pp_rank) + " next_point:" + str(
                   #             self.post_node_rank) + " send_next_point:" + str(self.pre_node_rank - 2))
                            # self.comm.send(current_micro_output.data, dst=dst, stream=cupy_send_stream)
                            self.comm.recv(self.input_micro_batches[i], src=src, stream=cupy_recv_stream)
                 #           print("global——rankID  completed  data computation ：" + str(self.global_rank))
                        else:
                            self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)

                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    # 如果是A100 执行concatTensor的计算，否则是T4
                    if self.device_gpu == 1:
                        current_micro_output = self.model(self.concatenated_tensor[i])
                    else:
                        # print("!!! T4 compute !  . pp_rank: "+str(self.pp_rank)+". global_rank: "+str(self.global_rank))

                        current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    # 发送正好和接收相反，
                    if self.gather_send:
                        # self.gather_group_size = get_gather_world_size()
                        # 非gather的A100节点，都是发送节点，修改gather的目的地节点
                        # if self.pp_rank != self.gather_group_size - 1:
                        self.gather_comm.gather(current_micro_output.data, gather_list=gather_data,
                                                dst=self.gather_group_size - 1,
                                                stream=cupy_send_stream)
                    #  scatter中的接收节点，只有A100节点发送
                    elif self.scatter_send:
                        # print("!!! compute sucess!  ScatterID:"+str(self.pp_rank_scatter)+". pp_rank: "+str(self.pp_rank))
                        self.scatter_group_size = get_scatter_world_size()
                        # 等于0 就是始发节点，需要将现有的数据进行分离在发送
                        # if self.pp_rank == 0:
                        # 将现有的Tensor进行按照batch进行拆分(0维度)，转成list[Tensor]
                        chunked_tensors = torch.chunk(current_micro_output.data, chunks=self.scatter_group_size - 1,
                                                      dim=0)
                        # 转换为List[torch.Tensor]
                        scatter_tensor_list = [split_tensor for split_tensor in chunked_tensors]
                        # 需要将0位置增加一维度（自己本身）
                        new_tensor = torch.zeros_like(scatter_tensor_list[0])
                        # 插入到0位置
                        scatter_tensor_list.insert(0, new_tensor)
                        self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=scatter_tensor_list, src=0,
                                                  stream=cupy_send_stream)
                        #print("!!! Scatter send !  ScatterID:"+str(self.pp_rank_scatter)+". pp_rank: "+str(self.pp_rank))
#                        assert args.world_size == args.data_group_size * args.pipeline_group_size
                    # 不用区分，因为不是gather或者Scatter后，说明后面的节点和当前节点的设备是一致的
                    # elif self.device_gpu =="A100":
                    #     self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    elif self.device_gpu==1:
               #         print("global——rankID  completed  data send (A100 model) ：" + str(self.global_rank))
                #        print("ID number :" + str(self.pp_rank) + " Global ID :" + str(self.global_rank)+ " next_point:" + str(self.post_node_rank))
                        self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    else:
                        if self.pipeline_group_size > 10:
                            dst = self.post_node_rank + 2
              #              print("ID number :" + str(self.pp_rank) + " next_point:" + str(
                           # self.post_node_rank) + " send_next_point:" + str(self.post_node_rank + 2))
                            self.comm.send(current_micro_output.data, dst=dst, stream=cupy_send_stream)
                        else:
             #               print("ID number :" + str(self.pp_rank) + " next_point:" + str(self.post_node_rank))
                            self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)
            output_micro_batches.append(current_micro_output)
        if self.enable_tidy_profiling:
            self.profiling_forward_stage()
        return output_micro_batches

    def profiling_forward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.first_node == False:
                recv_slot = self.forward_recv_start_events[i].elapsed_time(self.forward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_comp_start_events[i].elapsed_time(self.forward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

#            if self.pp_rank != self.pipeline_group_size - 1:# & self.pp_rank not in (11,12):

#                send_slot = self.forward_send_start_events[i].elapsed_time(self.forward_send_end_events[i]) * 1e+3
 #               send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
  #                          "ts": self.get_ts(self.forward_send_start_events[i]), "dur": send_slot,
   #                         "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
             #print(send_log)
    #            self.profiling_log.append(send_log)

    def _loss_compute(self, input_, target):
        # print(input_.shape, target.shape)
        if self.model.task == 'SeqClassification':
            return torch.nn.functional.cross_entropy(input=input_, target=target)
        elif self.model.task == 'Seq2SeqClassification':
            # shift_logits = input_[..., :-1, :].contiguous()
            # shift_labels = target[..., 1:].contiguous()
            # return torch.nn.functional.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return self.model(input_)

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None):
        # print("Backward stage start! rank-", self.rank) 还有42,47 节点
        if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            assert (target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert (target is None)
        self.gather_grad_recv = False
        self.gather_grad_send = False
        self.scatter_grad_send = False
        self.scatter_grad_recv = False
        if self.gather_comm is not None:
            self.pp_rank_gather = get_pipeline_gather_rank()
            # self.pp_rank_scatter  = get_pipeline_scatter_rank()
            self.gather_group_size = get_gather_world_size()
            if self.pp_rank_gather == self.gather_group_size - 1:
                # 变成了发送节点
                self.gather_grad_send = True
            else:
                # 接收节点
                self.gather_grad_recv = True
        if self.scatter_comm is not None:
            # self.pp_rank_gather = get_pipeline_gather_rank()
            self.pp_rank_scatter = get_pipeline_scatter_rank()
            self.scatter_group_size = get_scatter_world_size()
            if self.pp_rank_scatter == 0:
                self.scatter_grad_recv = True
            else:
                # 变成了发送节点
                self.scatter_grad_send = True
        for i in range(self.micro_batch_num):
            # 定义空的缓存区

            gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                           range(4)]
            scatter_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                            range(4)]
            if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:  # only send grad back to last node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_backward_comp_start(i)
                    if self.model.task == 'Seq2SeqClassification':
                        cached_output_micro_batches[i].backward()
                        print("backward Seq2Seq")
                    else:
                        # 计算loss和开始反向传播
                        if self.device_gpu == 1:
                            # A100 要匹配T4的标签
                            target = target_as_micro_batches[i].repeat(3)
                        else:
                            target = target_as_micro_batches[i]

                        loss = torch.nn.functional.cross_entropy(input=cached_output_micro_batches[i],
                                                                 target=target)
                        loss.backward()
                        if i%5==0:
                            print("micro_batch_num "+str(i)+", Loss is "+str(loss))# 0.9841
                    # print("list periphere:",len(scatter_tensor_list))
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    # scatter grade
                    # if i ==0:
                    #     #AttributeError: 'NoneType'
                    #     # 问题出在这里，
                    #     print(self.concatenated_tensor[i].requires_grad == True)
                    # 判断是否存在scatter或者gather，然后判断是否为A100，否则T4。
                    if self.gather_grad_send:
                        chunked_tensors = torch.chunk(self.concatenated_tensor[i].grad,
                                                      chunks=self.gather_group_size - 1, dim=0)

                        # 转换为List[torch.Tensor]
                        scatter_tensor_list = [split_tensor for split_tensor in chunked_tensors]
                        # scatter_tensor_list = [split_tensor for split_tensor in split_tensors]
                        new_tensor = torch.zeros_like(scatter_tensor_list[0])
                        # 添加在最后的位置
                        scatter_tensor_list.append(new_tensor)
                        # 用gather通讯租的Scatter方法
                        self.gather_comm.scatter(self.input_micro_batches[i].grad, scatter_list=scatter_tensor_list,
                                                 src=self.gather_group_size - 1, stream=cupy_send_stream)

                    elif self.scatter_grad_send:
                        #     Scatter的话，就采用聚合方法,向0节点进行发送
                        # scatter_tensor_list 表示空的列表
                        #     gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                        #                range(self.scatter_group_size)]
                        self.scatter_comm.gather(self.input_micro_batches[i].grad, gather_list=scatter_data, dst=0,
                                                 stream=cupy_send_stream)
                    elif self.device_gpu == 1:
                        # AttributeError: 'NoneType'

                        # print(self.concat_micro_batches_grad[i])# True
                        # print(self.concat_micro_batches_grad[i].is_leaf)# True
                        # print(self.concat_micro_batches_grad[i].grad is None)#True

                        self.comm.send(self.concatenated_tensor[i].grad, dst=self.pre_node_rank,
                                       stream=cupy_send_stream)
                    else:
                        if self.pipeline_group_size > 11:
                            dst = self.pre_node_rank - 2
                            self.comm.send(self.input_micro_batches[i].grad, dst=dst,stream=cupy_send_stream)
                        else:

                            self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank,stream=cupy_send_stream)

                    self.profile_mark_backward_send_end(i)
                # self.input_micro_batches[i].grad = None
                # torch.cuda.synchronize()  # Notice this for memory optimization
            elif self.first_node:  # only receive grad from previous node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    if self.gather_grad_recv:
                        #   执行对应的命令
                        self.gather_comm.scatter(self.output_micro_batches_grad[i], scatter_list=gather_data,
                                                 src=self.gather_group_size - 1, stream=cupy_recv_stream)
                    else:
                        self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank,
                                       stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    if self.device_gpu == 1:
                        cached_output_micro_batches[i].backward(gradient=self.concat_micro_batches_grad[i])
                    else:
                        cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:  # receive, compute and send zhongjianjiedian
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    if self.gather_grad_recv:
                        # gather接收是T4，仅接受即可
                        self.gather_comm.scatter(self.output_micro_batches_grad[i], scatter_list=gather_data,
                                                 src=self.gather_group_size - 1, stream=cupy_recv_stream)
                    #     对应的是A100，接收后需要处理
                    elif self.scatter_grad_recv:
                        self.scatter_comm.gather(self.output_micro_batches_grad[i], gather_list=scatter_data,
                                                 dst=self.pp_rank_scatter, stream=cupy_recv_stream)
                        scatter_data.pop(self.pp_rank_scatter)
                        self.concat_micro_batches_grad[i] = torch.cat(scatter_data, dim=0)
                        self.concat_micro_batches_grad[i].requires_grad_(True)
                    elif self.device_gpu == 1:
                        self.comm.recv(self.concat_micro_batches_grad[i], src=self.post_node_rank,
                                       stream=cupy_recv_stream)
                    else:
                        if self.pipeline_group_size > 11:
                            src = self.post_node_rank + 2
                            self.comm.recv(self.output_micro_batches_grad[i], src=src, stream=cupy_recv_stream)
                        else:
                            self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank,
                                           stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    if self.device_gpu == 1:
                        # print(self.concat_micro_batches_grad[i])
                        cached_output_micro_batches[i].backward(gradient=self.concat_micro_batches_grad[i])
                    else:
                        cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])

                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    # A100发送到子节点 将grad差分，然后使用gather通信组的Scatter
                    if self.gather_grad_send:
                        # if i ==0:
                        # AttributeError: 'NoneType'

                        # print(self.input_micro_batches[i].grad)
                        # print(self.input_micro_batches[i].grad is None)
                        # print(self.input_micro_batches[i].requires_grad == True)
                        # print(self.input_micro_batches[i].is_leaf)
                        # print("--------------------------------------------concat tensor-----------------------------------")
                        # print(self.concatenated_tensor[i].grad is None)
                        # 问题出在这里，
                        #    print(self.concat_micro_batches_grad[i].requires_grad == True)
                        #    print(self.concat_micro_batches_grad[i].grad)
                        # torch.Size([2, 2048, 2048]
                        # print("input shape :------------------------------------------"+str(self.concat_micro_batches_grad[i].grad.shape))
                        # print(self.concat_micro_batches_grad[i])# True
                        # print(self.concat_micro_batches_grad[i].is_leaf)# True
                        # print(self.concat_micro_batches_grad[i].grad is None)#True
                        chunked_tensors = torch.chunk(self.concatenated_tensor[i].grad,
                                                      chunks=self.gather_group_size - 1, dim=0)

                        # 转换为List[torch.Tensor]
                        scatter_tensor_list = [split_tensor for split_tensor in chunked_tensors]
                        # scatter_tensor_list = [split_tensor for split_tensor in split_tensors]
                        new_tensor = torch.zeros_like(scatter_tensor_list[0])
                        # 添加在最后的位置
                        scatter_tensor_list.append(new_tensor)
                        # 用gather通讯租的Scatter方法
                        self.gather_comm.scatter(self.input_micro_batches[i].grad, scatter_list=scatter_tensor_list,
                                                 src=self.pp_rank_gather, stream=cupy_send_stream)
                    # T4发送到A100节点 使用Scatter通信组的gather直接发送
                    elif self.scatter_grad_send:
                        self.scatter_comm.gather(self.input_micro_batches[i].grad, gather_list=scatter_data, dst=0,
                                                 stream=cupy_send_stream)
                    elif self.device_gpu == 1:
                        # if i ==0:
                        # print(self.concatenated_tensor[i].requires_grad == True)# False
                        # print(self.concatenated_tensor[i].grad)
                        # torch.Size([2, 2048, 2048]
                        # print("input shape :------------------------------------------"+str(self.concat_micro_batches_grad[i].grad.shape))
                        # print(self.concat_micro_batches_grad[i])# True
                        # print(self.concat_micro_batches_grad[i].is_leaf)# True
                        # print(self.concat_micro_batches_grad[i].grad is None)#True

                        self.comm.send(self.concatenated_tensor[i].grad, dst=self.pre_node_rank,
                                       stream=cupy_send_stream)

                    else:
                        if self.pipeline_group_size > 11:
                            dst = self.pre_node_rank - 2
                            self.comm.send(self.input_micro_batches[i].grad, dst=dst, stream=cupy_send_stream)
                        else:
                            self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank,
                                           stream=cupy_send_stream)
                    self.profile_mark_backward_send_end(i)
        if self.enable_tidy_profiling:
            self.profiling_backward_stage()

    def profiling_backward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            list_last = (20, 21, 22, 23, 24, 25)
            if self.pp_rank != self.pipeline_group_size - 1:
                if self.pipeline_group_size < 11:
                    recv_slot = self.backward_recv_start_events[i].elapsed_time(
                        self.backward_recv_ready_events[i]) * 1e+3
                    recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "4. backward-recv",
                                "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": recv_slot,
                                "args": {"micro-batch": i}, "cname": "startup"}
                    # print(recv_log)
                    self.profiling_log.append(recv_log)

            comp_slot = self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "5. backward-compute",
                        "ts": self.get_ts(self.backward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)
            if self.first_node == False:
                send_slot = self.backward_send_start_events[i].elapsed_time(self.backward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "6. backward-send",
                            "ts": self.get_ts(self.backward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def optimizer_step(self):
        if self.use_dp:
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.record_event(self.dp_optim.backward_ready_event)
            start = time.time()
            self.dp_optim.optimizer_step()
            endtime = time.time()
            print("dp_optim_spend_time:", endtime - start)
        else:
            with torch.cuda.stream(self.torch_comp_stream):
                if self.enable_tidy_profiling:
                    self.optimizer_start_event.record()
                # print("local optimizer")
                start = time.time()
                self.optimizer.step()
                # self.comm.barrier()
                endtime = time.time()
                print("local optimizer", endtime - start)
                if self.enable_tidy_profiling:
                    self.optimizer_end_event.record()
        if self.enable_tidy_profiling:
            self.profiling_optimizer_step()

    def profiling_optimizer_step(self):
        torch.cuda.synchronize()
        if not self.use_dp:
            optimizer_slot = self.optimizer_start_event.elapsed_time(self.optimizer_end_event) * 1e+3
            optimizer_log = {"name": "opt", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-step",
                             "ts": self.get_ts(self.optimizer_start_event), "dur": optimizer_slot, "cname": "bad"}
            # print(optimizer_log)
            self.profiling_log.append(optimizer_log)
        else:
            self.profiling_log.extend(self.dp_optim.profiling_data_parallel(self.init_time_stamp, self.init_event))

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def sgd_iter(self, input_=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()
        self.zero_input_grad()
        self.optimizer.zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            outputs = self.forward_stage(input_, target)
            forward_time = time.time()
            if step == 0:
                forward_slot = forward_time - start_time
            else:
                forward_slot = forward_time - backward_time
            print("Rank {} node forward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
            self.comm.barrier()  # This is an educated guess that such barrier would make it fair TC (probably required)
            self.backward_stage(outputs, target)
            backward_time = time.time()
            print("Rank {} node backward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, backward_time - forward_time))
        optimizer_time = time.time()
        self.optimizer_step()  # 15s
        optimizer_end_time = time.time()
        torch_synchronize_time = time.time()
        torch.cuda.synchronize()  # 0.5
        # optimizer_time = time.time()# 0.5s
        torch_syn_end_time = time.time()
        barrier_time = time.time()
        self.comm.barrier()
        barrier_end_time = time.time()
        end_time = time.time()
        print("                                                    Rank {} node optimizer step takes {:3.2f}s".format(
            self.global_rank, optimizer_end_time - optimizer_time))
        print(
            "                                                    Rank {} node torch_synchronize_time step takes {:3.2f}s".format(
                self.global_rank, torch_syn_end_time - torch_synchronize_time))
        print(
            "                                                    Rank {} node barrier_time step takes {:3.2f}s".format(
                self.global_rank, barrier_end_time - barrier_time))
        print(
            "                                                    Rank {} node optimizer step ALL(1+2+3) takes {:3.2f}s".format(
                self.global_rank, barrier_end_time - optimizer_time))
        iter_time = end_time - start_time
        print("                                                    Rank {} node whole iteration takes {:3.2f}s".format(
            self.global_rank, iter_time))
        print("-------------------------------------------")
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())

        return iter_time
