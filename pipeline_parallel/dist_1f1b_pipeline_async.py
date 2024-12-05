import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer


class Async1F1B:
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
        # self.use_dp = use_dp
        self.use_dp = use_dp
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.global_rank = args.rank
        self.pipeline_group_size = get_pipeline_parallel_world_size()
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1  # 第一个   的 前一个是-1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1  # 最后一个 的 后一个是-1
        self.comm_size = get_pipeline_parallel_world_size()
        self.comm = get_pipeline_parallel_comm()
        self.comm_b = get_pipeline_parallel_comm_b()
        # 获取gather结果
        self.gather_comm = get_pipeline_gather_comm()
        self.gather_comm_b = get_pipeline_gather_comm_b()

        self.scatter_comm = get_pipeline_scatter_comm()
        self.scatter_comm_b = get_pipeline_scatter_comm_b()

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
        # self.enable_tidy_profiling=False
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.torch_comp_stream_b = torch.cuda.Stream(device=device, priority=-1)
        self.torch_recv_stream_b = torch.cuda.Stream(device=device, priority=-2)
        self.torch_send_stream_b = torch.cuda.Stream(device=device, priority=-2)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]

        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        
        # self.enable_tidy_profiling=False
        # self.profiling_log = []

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

        # if self.device_gpu == 1:
        #     if self.pp_rank == 0:
        #         self.first_node = True
        # else:
        #     if self.pp_rank <= 2:
        #         self.first_node = True
        
        # if self.pp_rank == 0:
        #     self.first_node = True

        # TODO: 
        if self.pp_rank == 0:
                self.first_node = True
        

        # TODO: 
        # if self.first_node:
        #     self.input_micro_batches = None
        #     self.concatenated_tensor = None
        # else:
        
        
        # TODO: 
        # self.output_micro_batches = []
        self.current_micro_output = [None] * self.micro_batch_num
        
        
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
            # self.input_micro_batches = [None] * self.micro_batch_num
            # self.concatenated_tensor = [None] * self.micro_batch_num


        # if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
        #     self.output_micro_batches_grad = None
        # else:
        #     if self.device_gpu==0:
        #         self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
        #                                                   requires_grad=False, device=self.device, dtype=self.dtype)
        #                                       for _ in range(self.micro_batch_num)]
        #     else:
        #         self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
        #                                                   requires_grad=False, device=self.device, dtype=self.dtype)
        #                                       for _ in range(self.micro_batch_num)]
        #         self.concat_micro_batches_grad = [
        #             torch.zeros((self.micro_batch_size * 3, self.seq_length, self.embedding_dim),
        #                     requires_grad=False, device=self.device, dtype=self.dtype)
        #             for _ in range(self.micro_batch_num)]
        

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





        # TODO: Forward
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


        # Backward
        self.gather_grad_recv = False
        self.gather_grad_send = False
        self.scatter_grad_send = False
        self.scatter_grad_recv = False
        if self.gather_comm is not None:
            self.pp_rank_gather = get_pipeline_gather_rank()
            self.gather_group_size = get_gather_world_size()
            if self.pp_rank_gather == self.gather_group_size - 1:
                self.gather_grad_send = True
            else:
                self.gather_grad_recv = True
        if self.scatter_comm is not None:
            self.pp_rank_scatter = get_pipeline_scatter_rank()
            self.scatter_group_size = get_scatter_world_size()
            if self.pp_rank_scatter == 0:
                self.scatter_grad_recv = True
            else:
                self.scatter_grad_send = True

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
            self.torch_comp_stream_b.record_event(self.backward_comp_start_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream_b.record_event(self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream_b.record_event(self.backward_send_start_events[i])

    def profile_mark_backward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream_b.record_event(self.backward_send_end_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def _loss_compute(self, input_, target):
        # print(input_.shape, target.shape)
        if self.model.task == 'SeqClassification':
            return torch.nn.functional.cross_entropy(input=input_, target=target)
        elif self.model.task == 'Seq2SeqClassification':
            # shift_logits = input_[..., :-1, :].contiguous()
            # shift_labels = target[..., 1:].contiguous()
            # return torch.nn.functional.nll_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return self.model(input_)

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


    def forward_micro_batch(self, forward_index):
        # print("Forward stage start! rank-", self.rank)    
        
        # output_micro_batches = []
        
        i = forward_index


        # for i in range(self.micro_batch_num):
        gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in range(4)]

        if self.first_node:
            
            with torch.cuda.stream(self.torch_comp_stream):
                self.profile_mark_forward_comp_start(i)
                # 如果是A100要复制3份数据
                if self.device_gpu==1:
                    print(self.input_micro_batches[i].shape)
                    # TODO: repeat
                    self.concatenated_tensor[i] = self.input_micro_batches[i].repeat(3,1)
                    self.current_micro_output[i] = self.model(self.concatenated_tensor[i])
                    # current_micro_output = self.model(self.concatenated_tensor[i])
                    print(f"Forward -- Current index i: {i} compute, fisrt node, A100")
                else:
                    # current_micro_output = self.model(self.input_micro_batches[i])
                    # 1 1 0; 1 1 1
                    # print(f"Length of self.input_micro_batches: {len(self.input_micro_batches)}")
                    # print(f"Length of self.current_micro_output: {len(self.current_micro_output)}")
                    # print(f"Current index i: {i}")
                    self.current_micro_output[i] = self.model(self.input_micro_batches[i])
                    print(f"Forward -- Current index i: {i} compute, fisrt node, T4")
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                self.profile_mark_forward_send_start(i)
                if self.gather_comm is not None:
                    self.gather_group_size = get_gather_world_size()
                    self.gather_comm.gather(self.current_micro_output[i].data, gather_list=gather_data,
                                            dst=self.gather_group_size - 1, stream=cupy_send_stream)
                else:
                    self.comm.send(self.current_micro_output[i].data, dst=self.post_node_rank, stream=cupy_send_stream)
                self.profile_mark_forward_send_end(i)


        elif self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_forward_recv_start(i)

                if self.gather_recv:
                    self.gather_comm.gather(self.input_micro_batches[i], gather_list=gather_data,
                                            dst=self.gather_group_size - 1, stream=cupy_recv_stream)
                    # 方案1 将list数组使用torch的concat进行合并
                    gather_data.pop(self.pp_rank_gather)
                    self.concatenated_tensor[i] = torch.cat(gather_data, dim=0)
                    self.concatenated_tensor[i].requires_grad_(True)
                elif self.scatter_recv:
                    self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=gather_data, src=0,
                                                stream=cupy_recv_stream)
                elif self.device_gpu == 1:
                    self.comm.recv(self.concatenated_tensor[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                else:
                    if self.pipeline_group_size > 10:
                        src = self.pre_node_rank - 2
                        self.comm.recv(self.input_micro_batches[i], src=src, stream=cupy_recv_stream)
                    else:
                        self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
            
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                self.profile_mark_forward_comp_start(i)
                if self.model.task == 'Seq2SeqClassification':
                    # current_micro_output = self.model(self.input_micro_batches[i], target_data_micro_batches[i])
                    print('先不管这个任务')
                else:
                    if self.device_gpu == 1:
                        # current_micro_output = self.model(self.concatenated_tensor[i])
                        self.current_micro_output[i] = self.model(self.concatenated_tensor[i])
                        print(f"Forward -- Current index i: {i} compute, last node, A100")
                        
                    else:
                        # current_micro_output = self.model(self.input_micro_batches[i])
                        self.current_micro_output[i] = self.model(self.input_micro_batches[i])
                        print(f"Forward -- Current index i: {i} compute, last node, T4")
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
        
        
        else:
            
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.profile_mark_forward_recv_start(i)
                # gather接收节点，只有最后的目的节点接收，其他的都是发送节点
                if self.gather_recv:
                    self.gather_comm.gather(self.input_micro_batches[i], gather_list=gather_data,
                                            dst=self.gather_group_size - 1, stream=cupy_recv_stream)
                    # gather中的A100节点需要对聚合数据进行处理
                    gather_data.pop(self.pp_rank_gather)
                    self.concatenated_tensor[i] = torch.cat(gather_data, dim=0)
                    self.concatenated_tensor[i].requires_grad_(True)
                elif self.scatter_recv:
                    self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=gather_data, src=0,
                                                stream=cupy_recv_stream)
                elif self.device_gpu == 1:
                    self.comm.recv(self.concatenated_tensor[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                else:
                    # 否则为T4节点
                    if self.pipeline_group_size > 10:
                        src = self.pre_node_rank - 2
                        self.comm.recv(self.input_micro_batches[i], src=src, stream=cupy_recv_stream)
                    else:
                        self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
            
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                self.profile_mark_forward_comp_start(i)
                if self.device_gpu == 1:
                    # current_micro_output = self.model(self.concatenated_tensor[i])
                    self.current_micro_output[i] = self.model(self.concatenated_tensor[i])
                    print(f"Forward -- Current index i: {i} compute, middle node, A100")
                    
                else:
                    # current_micro_output = self.model(self.input_micro_batches[i])
                    self.current_micro_output[i] = self.model(self.input_micro_batches[i])
                    print(f"Forward -- Current index i: {i} compute, middle node, T4")
                self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                self.profile_mark_forward_send_start(i)
                # 发送正好和接收相反，
                if self.gather_send:
                    self.gather_comm.gather(self.current_micro_output[i].data, gather_list=gather_data,
                                            dst=self.gather_group_size - 1,
                                            stream=cupy_send_stream)
                elif self.scatter_send:
                    self.scatter_group_size = get_scatter_world_size()
                    # 等于0 就是始发节点，需要将现有的数据进行分离在发送
                    # if self.pp_rank == 0:
                    # 将现有的Tensor进行按照batch进行拆分(0维度)，转成list[Tensor]
                    chunked_tensors = torch.chunk(self.current_micro_output[i].data, chunks=self.scatter_group_size - 1,
                                                    dim=0)
                    # 转换为List[torch.Tensor]
                    scatter_tensor_list = [split_tensor for split_tensor in chunked_tensors]
                    # 需要将0位置增加一维度（自己本身）
                    new_tensor = torch.zeros_like(scatter_tensor_list[0])
                    # 插入到0位置
                    scatter_tensor_list.insert(0, new_tensor)
                    self.scatter_comm.scatter(self.input_micro_batches[i], scatter_list=scatter_tensor_list, src=0,
                                                stream=cupy_send_stream)

                elif self.device_gpu==1:
                    self.comm.send(self.current_micro_output[i].data, dst=self.post_node_rank, stream=cupy_send_stream)
                else:
                    if self.pipeline_group_size > 10:
                        dst = self.post_node_rank + 2
                        self.comm.send(self.current_micro_output[i].data, dst=dst, stream=cupy_send_stream)
                    else:
                        self.comm.send(self.current_micro_output[i].data, dst=self.post_node_rank, stream=cupy_send_stream)
                self.profile_mark_forward_send_end(i)
            
            # output_micro_batches.append(current_micro_output)
        
        # if self.enable_tidy_profiling:
        #     self.profiling_forward_stage()
        # return output_micro_batches
    
    
    def backward_micro_batch(self, backward_index,cached_output_micro_batches: List[torch.Tensor], target_as_micro_batches=None):
        # print("Backward stage start! rank-", self.rank)
        
        
        i = backward_index
        
        # for i in range(self.micro_batch_num):
        # 定义空的缓存区
        
        gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                        range(4)]
        scatter_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                        range(4)]
        
        
        if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:  # only send grad back to last node, do not receive
            
            with torch.cuda.stream(self.torch_comp_stream_b):
                self.profile_mark_backward_comp_start(i)
                if self.model.task == 'Seq2SeqClassification':
                    cached_output_micro_batches[i].backward()
                    print("backward Seq2Seq")
                else:
                    # 计算loss和开始反向传播
                    if self.device_gpu == 1:
                        # A100 要匹配T4的标签
                        # TODO: repeat31
                        target = target_as_micro_batches[i].repeat(3)
                    else:
                        target = target_as_micro_batches[i]

                    loss = torch.nn.functional.cross_entropy(input=cached_output_micro_batches[i],
                                                                target=target)
                    loss.backward()
                    print(f"\t\tBackward -- Current index i: {i} compute, last node")
                    if i%5==0:
                        print("micro_batch_num "+str(i)+", Loss is "+str(loss))# 0.9841
                self.torch_comp_stream_b.record_event(self.backward_comp_ready_events[i])
            
            with torch.cuda.stream(self.torch_send_stream_b):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream_b.cuda_stream)
                self.torch_send_stream_b.wait_event(self.backward_comp_ready_events[i])
                self.profile_mark_backward_send_start(i)
                
                if self.gather_grad_send:
                    chunked_tensors = torch.chunk(self.concatenated_tensor[i].grad,
                                                    chunks=self.gather_group_size - 1, dim=0)

                    # 转换为List[torch.Tensor]
                    scatter_tensor_list = [split_tensor for split_tensor in chunked_tensors]
                    # scatter_tensor_list = [split_tensor for split_tensor in split_tensors]
                    new_tensor = torch.zeros_like(scatter_tensor_list[0])
                    # 添加在最后的位置
                    scatter_tensor_list.append(new_tensor)
                    # 用gather通讯组的Scatter方法
                    self.gather_comm_b.scatter(self.input_micro_batches[i].grad, scatter_list=scatter_tensor_list,
                                                src=self.gather_group_size - 1, stream=cupy_send_stream)

                elif self.scatter_grad_send:
                    # scatter的话，就采用聚合方法,向0节点进行发送
                    # scatter_tensor_list 表示空的列表
                    #     gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                    #                range(self.scatter_group_size)]
                    self.scatter_comm_b.gather(self.input_micro_batches[i].grad, gather_list=scatter_data, dst=0,
                                                stream=cupy_send_stream)
                elif self.device_gpu == 1:
                    self.comm_b.send(self.concatenated_tensor[i].grad, dst=self.pre_node_rank,
                                    stream=cupy_send_stream)
                else:
                    if self.pipeline_group_size > 11:
                        dst = self.pre_node_rank - 2
                        self.comm_b.send(self.input_micro_batches[i].grad, dst=dst,stream=cupy_send_stream)
                    else:

                        self.comm_b.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank,stream=cupy_send_stream)

                self.profile_mark_backward_send_end(i)

        elif self.first_node:  # only receive grad from previous node, do not send
            with torch.cuda.stream(self.torch_recv_stream_b):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream_b.cuda_stream)
                self.profile_mark_backward_recv_start(i)
                if self.gather_grad_recv:
                    #   执行对应的命令
                    self.gather_comm_b.scatter(self.output_micro_batches_grad[i], scatter_list=gather_data,
                                                src=self.gather_group_size - 1, stream=cupy_recv_stream)
                else:
                    self.comm_b.recv(self.output_micro_batches_grad[i], src=self.post_node_rank,
                                    stream=cupy_recv_stream)
                self.torch_recv_stream_b.record_event(self.backward_recv_ready_events[i])
            
            with torch.cuda.stream(self.torch_comp_stream_b):
                self.torch_comp_stream_b.wait_event(self.backward_recv_ready_events[i])
                self.profile_mark_backward_comp_start(i)
                
                if self.device_gpu == 1:
                    cached_output_micro_batches[i].backward(gradient=self.concat_micro_batches_grad[i])
                    print(f"\t\tBackward -- Current index i: {i} compute, first node, A100")
                else:
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    print(f"\t\tBackward -- Current index i: {i} compute, first node, T4")
                self.torch_comp_stream_b.record_event(self.backward_comp_ready_events[i])
        
        else:  # receive, compute and send zhongjianjiedian
            with torch.cuda.stream(self.torch_recv_stream_b):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream_b.cuda_stream)
                self.profile_mark_backward_recv_start(i)
                if self.gather_grad_recv:
                    # gather接收是T4，仅接受即可
                    self.gather_comm_b.scatter(self.output_micro_batches_grad[i], scatter_list=gather_data,
                                                src=self.gather_group_size - 1, stream=cupy_recv_stream)
                    # 对应的是A100，接收后需要处理
                elif self.scatter_grad_recv:
                    self.scatter_comm_b.gather(self.output_micro_batches_grad[i], gather_list=scatter_data,
                                                dst=self.pp_rank_scatter, stream=cupy_recv_stream)
                    scatter_data.pop(self.pp_rank_scatter)
                    self.concat_micro_batches_grad[i] = torch.cat(scatter_data, dim=0)
                    self.concat_micro_batches_grad[i].requires_grad_(True)
                elif self.device_gpu == 1:
                    self.comm_b.recv(self.concat_micro_batches_grad[i], src=self.post_node_rank,
                                    stream=cupy_recv_stream)
                else:
                    if self.pipeline_group_size > 11:
                        src = self.post_node_rank + 2
                        self.comm_b.recv(self.output_micro_batches_grad[i], src=src, stream=cupy_recv_stream)
                    else:
                        self.comm_b.recv(self.output_micro_batches_grad[i], src=self.post_node_rank,
                                        stream=cupy_recv_stream)
                self.torch_recv_stream_b.record_event(self.backward_recv_ready_events[i])
            
            with torch.cuda.stream(self.torch_comp_stream_b):
                self.torch_comp_stream_b.wait_event(self.backward_recv_ready_events[i])
                self.profile_mark_backward_comp_start(i)
                
                if self.device_gpu == 1:
                    # print(self.concat_micro_batches_grad[i])
                    cached_output_micro_batches[i].backward(gradient=self.concat_micro_batches_grad[i])
                    print(f"\t\tBackward -- Current index i: {i} compute, middle node, A100")
                    
                else:
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    print(f"\t\tBackward -- Current index i: {i} compute, middle node, T4")
                    

                self.torch_comp_stream_b.record_event(self.backward_comp_ready_events[i])
            
            with torch.cuda.stream(self.torch_send_stream_b):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream_b.cuda_stream)
                self.torch_send_stream_b.wait_event(self.backward_comp_ready_events[i])
                self.profile_mark_backward_send_start(i)
                # A100发送到子节点 将grad差分，然后使用gather通信组的Scatter
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
                    self.gather_comm_b.scatter(self.input_micro_batches[i].grad, scatter_list=scatter_tensor_list,
                                                src=self.pp_rank_gather, stream=cupy_send_stream)
                # T4发送到A100节点 使用Scatter通信组的gather直接发送
                elif self.scatter_grad_send:
                    self.scatter_comm_b.gather(self.input_micro_batches[i].grad, gather_list=scatter_data, dst=0,
                                                stream=cupy_send_stream)
                elif self.device_gpu == 1:
                    self.comm_b.send(self.concatenated_tensor[i].grad, dst=self.pre_node_rank,
                                    stream=cupy_send_stream)
                else:
                    if self.pipeline_group_size > 11:
                        dst = self.pre_node_rank - 2
                        self.comm_b.send(self.input_micro_batches[i].grad, dst=dst, stream=cupy_send_stream)
                    else:
                        self.comm_b.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank,
                                        stream=cupy_send_stream)
                self.profile_mark_backward_send_end(i)
    
        # if self.enable_tidy_profiling:
        #     self.profiling_backward_stage()
            
            
    def forward_backward_stages(self, input_data=None, target=None, loss_func=torch.nn.functional.cross_entropy):
        # this loading part should be updated later
        # if self.pp_rank == 0:
        #     assert(input_data is not None)
        #     self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        #     target_as_micro_batches = [None for _ in range(self.micro_batch_num)]
        # elif self.pp_rank == self.pipeline_group_size - 1:
        #     assert (target is not None)
        #     target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        # else:
        #     assert (input_data is None and target is None)
        #     target_as_micro_batches = [None for _ in range(self.micro_batch_num)]

        # TODO: forward
        if self.first_node:
            assert (input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)

        elif self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            if self.model.task == 'Seq2SeqClassification':
                # assert target_data is not None
                # target_data_micro_batches = torch.chunk(target_data, self.micro_batch_num, dim=0)
                print('用的另一个任务，先不管了！！！')

        # Backward
        if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            assert (target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert (target is None)
            target_as_micro_batches = [None for _ in range(self.micro_batch_num)]
        
        
        forward_i = 0
        backward_i = 0
        
        # TODO: 手动映射一下
        mapping =  [0, 0, 0, 1, 2, 3, 3, 3, 4]
        mapping_rank = mapping[self.pp_rank]
        
        print(f"pp_rank: {self.pp_rank}, mapping_rank: {mapping_rank}, forward compute micro-batches: {4 - mapping_rank}")

        # TODO: 这里的 pp_rank 需要改
        # Starting phase: to fill the pipeline_parallel.
        print('\n================================================================== start phase ==================================================================')
        while forward_i < self.pipeline_group_size - 1 - self.pp_rank:
        # while forward_i < 4 - mapping_rank:
        # while forward_i < 4:
            self.forward_micro_batch(forward_index=forward_i)
            forward_i += 1
        
        # Running phase: 1 forward coupled with 1 backward.
        print('\n================================================================== running phase ==================================================================')
        while forward_i < self.micro_batch_num:
            self.forward_micro_batch(forward_index=forward_i)
            self.backward_micro_batch(backward_index=backward_i,
                                      cached_output_micro_batches=self.current_micro_output,
                                      target_as_micro_batches=target_as_micro_batches)
            forward_i += 1
            backward_i += 1


        # Ending phase: to finish the rest stages in the pipeline_parallel.
        print('\n================================================================== ending phase ==================================================================')
        while backward_i < self.micro_batch_num:
            self.backward_micro_batch(backward_index=backward_i,
                                      cached_output_micro_batches=self.current_micro_output,
                                      target_as_micro_batches=target_as_micro_batches)
            backward_i += 1
        
        if self.enable_tidy_profiling:
            self.profile_forward_backward_stages()


    def profile_forward_backward_stages(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            # TODO: 修改一下
            # if self.pp_rank != 0:
            if not self.first_node:
                forward_recv_slot = \
                    self.forward_recv_start_events[i].elapsed_time(self.forward_recv_ready_events[i]) * 1e+3
                forward_recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                                    "ts": self.get_ts(self.forward_recv_start_events[i]), "dur": forward_recv_slot,
                                    "args": {"micro-batch": i}, "cname": "startup"}
                # print(forward_recv_log)

                self.profiling_log.append(forward_recv_log)

            forward_comp_slot = \
                self.forward_comp_start_events[i].elapsed_time(self.forward_comp_ready_events[i]) * 1e+3
            forward_comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                                "ts": self.get_ts(self.forward_comp_start_events[i]), "dur": forward_comp_slot,
                                "args": {"micro-batch": i}, "cname": "good"}
            # print(forward_comp_log)
            self.profiling_log.append(forward_comp_log)

            # if self.pp_rank != self.pipeline_group_size - 1:
            if not (self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11):

                forward_send_slot = \
                    self.forward_send_start_events[i].elapsed_time(self.forward_send_end_events[i]) * 1e+3
                forward_send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                                    "ts": self.get_ts(self.forward_send_start_events[i]), "dur": forward_send_slot,
                                    "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(forward_send_log)
                self.profiling_log.append(forward_send_log)

        for i in range(self.micro_batch_num):
            # if self.pp_rank != self.pipeline_group_size - 1:
            if not (self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11):
            
                backward_recv_slot = \
                    self.backward_recv_start_events[i].elapsed_time(self.backward_recv_ready_events[i]) * 1e+3
                backward_recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "4. backward-recv",
                                     "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": backward_recv_slot,
                                     "args": {"micro-batch": i}, "cname": "startup"}
                # print(backward_recv_log)
                self.profiling_log.append(backward_recv_log)

            backward_comp_slot = \
                self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            backward_comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "5. backward-compute",
                                 "ts": self.get_ts(self.backward_comp_start_events[i]),
                                 "dur": backward_comp_slot, "args": {"micro-batch": i}, "cname": "good"}
            # print(backward_comp_log)
            self.profiling_log.append(backward_comp_log)

            # if self.pp_rank != 0:
            if not self.first_node:
                backward_send_slot = \
                    self.backward_send_start_events[i].elapsed_time(self.backward_send_end_events[i]) * 1e+3
                backward_send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "6. backward-send",
                                     "ts": self.get_ts(self.backward_send_start_events[i]), "dur": backward_send_slot,
                                     "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(backward_send_log)
                self.profiling_log.append(backward_send_log)

    def sgd_iter(self, input_=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()
        
        self.zero_input_grad()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.forward_backward_stages(input_data=input_, target=target)
        self.optimizer_step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node 1f1b iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        return iter_time
