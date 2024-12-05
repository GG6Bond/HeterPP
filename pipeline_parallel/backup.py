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
        
        # 可以移出去
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

            if self.first_node:
                
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    # 如果是A100要复制3份数据
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
                        current_micro_output = self.model(self.input_micro_batches[i], target_data_micro_batches[i])
                    else:
                        if self.device_gpu == 1:
                            current_micro_output = self.model(self.concatenated_tensor[i])
                        else:
                            current_micro_output = self.model(self.input_micro_batches[i])
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
                    # 如果是A100 执行concatTensor的计算，否则是T4
                    if self.device_gpu == 1:
                        current_micro_output = self.model(self.concatenated_tensor[i])
                    else:
                        current_micro_output = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    # 发送正好和接收相反，
                    if self.gather_send:
                        self.gather_comm.gather(current_micro_output.data, gather_list=gather_data,
                                                dst=self.gather_group_size - 1,
                                                stream=cupy_send_stream)
                    elif self.scatter_send:
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

                    elif self.device_gpu==1:
                        self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    else:
                        if self.pipeline_group_size > 10:
                            dst = self.post_node_rank + 2
                            self.comm.send(current_micro_output.data, dst=dst, stream=cupy_send_stream)
                        else:
                            self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)
            
            # output_micro_batches.append(current_micro_output)
        
        if self.enable_tidy_profiling:
            self.profiling_forward_stage()
        # return output_micro_batches
        
        
        
def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None):
        # print("Backward stage start! rank-", self.rank)
        
        if self.pp_rank == self.pipeline_group_size - 1 or self.pp_rank >= 11:
            assert (target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert (target is None)
        
        # 可以提出去
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
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
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
                        self.gather_comm.scatter(self.input_micro_batches[i].grad, scatter_list=scatter_tensor_list,
                                                 src=self.gather_group_size - 1, stream=cupy_send_stream)

                    elif self.scatter_grad_send:
                        # scatter的话，就采用聚合方法,向0节点进行发送
                        # scatter_tensor_list 表示空的列表
                        #     gather_data = [torch.zeros_like(self.input_micro_batches[i]) for _ in
                        #                range(self.scatter_group_size)]
                        self.scatter_comm.gather(self.input_micro_batches[i].grad, gather_list=scatter_data, dst=0,
                                                 stream=cupy_send_stream)
                    elif self.device_gpu == 1:
                        self.comm.send(self.concatenated_tensor[i].grad, dst=self.pre_node_rank,
                                       stream=cupy_send_stream)
                    else:
                        if self.pipeline_group_size > 11:
                            dst = self.pre_node_rank - 2
                            self.comm.send(self.input_micro_batches[i].grad, dst=dst,stream=cupy_send_stream)
                        else:

                            self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank,stream=cupy_send_stream)

                    self.profile_mark_backward_send_end(i)

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
                        # 对应的是A100，接收后需要处理
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