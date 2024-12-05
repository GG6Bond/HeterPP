from .nccl_backend import *

_DATA_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = None
_DATA_PARALLEL_WORLD_SIZE = None

_PIPELINE_PARALLEL_COMM = None
_PIPELINE_PARALLEL_RANK = None
_PIPELINE_PARALLEL_WORLD_SIZE = None

#PTAFM
_PIPELINE_PARALLEL_SCATTER_COMM =None
_PIPELINE_PARALLEL_GATHER_COMM =None

_PIPELINE_PARALLEL_SCATTER_COMM_B =None
_PIPELINE_PARALLEL_GATHER_COMM_B =None
_PIPELINE_PARALLEL_COMM_B = None

_GATHE_WORLD_SIZE = None
_SCATTER_WORLD_SIZE = None


_DEVICE_GPU = None

_PIPELINE_GATHER_RANK =None
_PIPELINE_SCATTER_RANK =None

# =============================================== stage节点数（12 4 4 12 4）

# rank_A100 = [12,13,14,15,16,17,18,19,32,33,34,35]

# data_parallel_config = [[0,1,2,3,4,5,6,7,8,9,10,11],
#                         [12,15,18,21],
#                         [24,27,30,33],
#                         [36,37,38,39,40,41,42,43,44,45,46,47],
#                         [48,51,54,57]]

# pipeline_config = [[0,1,2,12,24,36,37,38,48],
#                    [3,4,5,15,27,39,40,41,51],
#                    [6,7,8,18,30,42,43,44,54],
#                    [9,10,11,21,33,45,46,47,57]]

# rank_mapping_id =[0,1,2,3,4,5,6,7,8,9,10,11,12,15,18,21,24,27,30,33,36,37,38,39,40,41,42,43,44,45,46,47,48,51,54,57]


# gathers_comm = [[0,1,2,12],
#                 [3,4,5,15],
#                 [6,7,8,18],
#                 [9,10,11,21],
#                 [36,37,38,48],
#                 [39,40,41,51],
#                 [42,43,44,54],
#                 [45,46,47,57]]

# scatters_comm = [[24,36,37,38],
#                  [27,39,40,41],
#                  [30,42,43,44],
#                  [33,45,46,47]]


# # ==========================================AAAAAAAA 异构 每个stage节点数（3 1 1 3 1）
# rank_A100 = [3,4,8]

# data_parallel_config = [[0,1,2],
#                         [3],
#                         [6],
#                         [9,10,11],
#                         [12]]

# pipeline_config = [[0,1,2,3,6,9,10,11,12]]

# rank_mapping_id =[0,1,2,3,6,9,10,11,12]


# gathers_comm = [[0,1,2,3],
#                 [9,10,11,12]]

# scatters_comm = [[6,9,10,11]]


# ==========================================BBBBBBB 异构 每个stage节点数（1 1 1 3 3）
# rank_A100 = [0,1,2]

# data_parallel_config = [[0],
#                         [3],
#                         [6],
#                         [9,10,11],
#                         [12,13,14]]

# pipeline_config = [[0,3,6,9,10,11,12,13,14]]

# rank_mapping_id =[0,3,6,9,10,11,12,13,14]


# gathers_comm = []

# scatters_comm = [[6,9,10,11]]


# ==========================================CCCCCC 异构 每个stage节点数（1 1 1 3 1）
# rank_A100 = [0,1,2,6]

# data_parallel_config = [[0],
#                         [3],
#                         [6],
#                         [9,10,11],
#                         [12]]

# pipeline_config = [[0,3,6,9,10,11,12]]

# rank_mapping_id =[0,3,6,9,10,11,12]


# gathers_comm = [[9,10,11,12]]

# scatters_comm = [[6,9,10,11]]


# =========================================== 6卡T4，同构

# rank_A100 = []

# rank_mapping_id =[0,1,2,3,4,5]


# data_parallel_config = [[0,1],
#                         [2,3],
#                         [4,5]]

# pipeline_config = [[0,2,4],
#                    [1,3,5]]

# gathers_comm = []

# scatters_comm = []

# =================================================



# # 消融实验 gather 和 scatter（全部节点）
# rank_A100 = []

# rank_mapping_id =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]



# data_parallel_config = [[0,1,2,3,4,5,6,7],
#                         [8,9,10,11,12,13,14,15],
#                         [16,17,18,19,20,21,22,23],
#                         [24,25,26,27,28,29,30,31],
#                         [32,33,34,35,36,37,38,39]]

# pipeline_config = [[0,8,16,24,32],
#                    [1,9,17,25,33],
#                    [2,10,18,26,34],
#                    [3,11,19,27,35],
#                    [4,12,20,28,36],
#                    [5,13,21,29,37],
#                    [6,14,22,30,38],
#                    [7,15,23,31,39]]

# gathers_comm = []

# scatters_comm = []

# 消融实验 gather 和 scatter（9 个节点）
# rank_A100 = []

# rank_mapping_id =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]



# data_parallel_config = [[0,1,2],
#                         [3,4,5],
#                         [6,7,8],
#                         [9,10,11],
#                         [12,13,14]]

# pipeline_config = [[0,3,6,9,12],
#                    [1,4,7,10,13],
#                    [2,5,8,11,14]]

# gathers_comm = []

# scatters_comm = []

# # 全部T4
# rank_A100 = []

# rank_mapping_id =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]



# data_parallel_config = [[0,1,2,3,4],
#                         [5,6,7,8,9],
#                         [10,11,12,13,14],
#                         [15,16,17,18,19],
#                         [20,21,22,23,24]]

# pipeline_config = [[0,5,10,15,20],
#                    [1,6,11,16,21],
#                    [2,7,12,17,22],
#                    [3,8,13,18,23],
#                    [4,9,14,19,24]]

# gathers_comm = []

# scatters_comm = []

# 全部A100
# rank_A100 = [0,1,2,3,4,5,6,7,8,9]


# rank_mapping_id =[0,1,2,3,4,5,6,7,8,9]



# data_parallel_config = [[0,1],
#                         [2,3],
#                         [4,5],
#                         [6,7],
#                         [8,9]]

# pipeline_config = [[0,2,4,6,8],
#                    [1,3,5,7,9]]

# gathers_comm = []

# scatters_comm = []


rank_A100 = []


rank_mapping_id =[0,1,2,3,4,5,6,7,8,9]



data_parallel_config = [[0,5],
                        [1,6],
                        [2,7],
                        [3,8],
                        [4,9]]

pipeline_config = [[0,1,2,3,4],
                   [5,6,7,8,9]]

gathers_comm = []

scatters_comm = []


def find_list(id, config):
    for i, sublist in enumerate(config):
        if id in sublist:
            return i, len(sublist)
    return None, None
def get_device_gpu() -> int:
    return _DEVICE_GPU
def get_pipeline_gather_rank() -> int:
    assert _PIPELINE_GATHER_RANK is not None
    return _PIPELINE_GATHER_RANK

def get_pipeline_scatter_rank() -> int:
    assert _PIPELINE_SCATTER_RANK is not None
    return _PIPELINE_SCATTER_RANK


def get_gather_world_size() -> int:
    assert _GATHE_WORLD_SIZE is not None
    return _GATHE_WORLD_SIZE
def get_scatter_world_size() -> int:
    assert _SCATTER_WORLD_SIZE is not None
    return _SCATTER_WORLD_SIZE

def get_pipeline_gather_comm() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_GATHER_COMM is not None
    return _PIPELINE_PARALLEL_GATHER_COMM

def get_pipeline_scatter_comm() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_SCATTER_COMM is not None
    return _PIPELINE_PARALLEL_SCATTER_COMM

def get_pipeline_gather_comm_b() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_GATHER_COMM is not None
    return _PIPELINE_PARALLEL_GATHER_COMM_B

def get_pipeline_scatter_comm_b() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_SCATTER_COMM is not None
    return _PIPELINE_PARALLEL_SCATTER_COMM_B

def get_data_parallel_comm() -> NCCLCommunicator:
    assert _DATA_PARALLEL_COMM is not None
    return _DATA_PARALLEL_COMM


def get_data_parallel_rank() -> int:
    assert _DATA_PARALLEL_RANK is not None
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    assert _DATA_PARALLEL_WORLD_SIZE is not None
    return _DATA_PARALLEL_WORLD_SIZE


def get_pipeline_parallel_comm() -> NCCLCommunicator:
    assert _PIPELINE_PARALLEL_COMM is not None
    return _PIPELINE_PARALLEL_COMM
def get_pipeline_parallel_comm_b() -> NCCLCommunicator:
    assert _PIPELINE_PARALLEL_COMM_B is not None
    return _PIPELINE_PARALLEL_COMM_B

def get_pipeline_parallel_rank() -> int:
    assert _PIPELINE_PARALLEL_RANK is not None
    return _PIPELINE_PARALLEL_RANK


def get_pipeline_parallel_world_size() -> int:
    assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
    return _PIPELINE_PARALLEL_WORLD_SIZE


def init_communicators(args):
    default_init(args)
    #print("1")
    #real_node_count = args.world_size+1
    #if args.rank==2:

    #args.world_size=args.world_size+1
    #assert args.world_size == args.data_group_size * args.pipeline_group_size
    #print("2") 
    print(f"{args.world_size} == {args.data_group_size} * {args.pipeline_group_size}")
    if args.data_group_size != args.data_group_size * args.pipeline_group_size:
        #    We do the following hard code alignment of communication groups:
        #    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
        #    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
        #    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
        #    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
        #assert args.world_size == args.data_group_size * args.pipeline_group_size
        global _DATA_PARALLEL_COMM
        global _PIPELINE_PARALLEL_COMM
        global _PIPELINE_PARALLEL_COMM_B
        global _DATA_PARALLEL_RANK
        global _PIPELINE_PARALLEL_RANK
        global _DATA_PARALLEL_WORLD_SIZE
        global _PIPELINE_PARALLEL_WORLD_SIZE
       
        # gather & scatter

        global _GATHE_WORLD_SIZE
        global _SCATTER_WORLD_SIZE
        global _DEVICE_GPU

        global _PIPELINE_GATHER_RANK
        global _PIPELINE_SCATTER_RANK
        # print(args.rank,'++++')
        mapping_rank = rank_mapping_id[args.rank]
        if args.rank in rank_A100:
            _DEVICE_GPU = 1
            print("A100 Node"+str(args.rank)+". MappingID:"+str(mapping_rank))
        else:
            print("T4 Node"+str(args.rank)+". MappingID:"+str(mapping_rank))
            _DEVICE_GPU=0
        # We use pipeline parallel by default.
        _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size
        #_PIPELINE_PARALLEL_RANK = args.rank % args.pipeline_group_size
        _PIPELINE_PARALLEL_RANK = args.rank
        # rank --> id mapping   21--->36
        #mapping_rank = rank_mapping_id[args.rank]
        # build pipellel group 0 11 [0,1,2,3,4,5,18,36,54,72,90]
        list_index,group_size = find_list(mapping_rank, pipeline_config)
        # [0,1,2,3,4,5,18,36,54,72,90]
        pipeline_list = pipeline_config[list_index]
        # 
        print("rankid:"+str(_PIPELINE_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(pipeline_config[list_index])+",group size:"+str(group_size))

        # id ----> rank 36----->7
        for i in range(group_size):
            if pipeline_list[i] == mapping_rank:
                _PIPELINE_PARALLEL_RANK = i
        print("_PIPELINE_PARALLEL_RANK:",_PIPELINE_PARALLEL_RANK)
        args.pipeline_group_size= group_size
        _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size

        _PIPELINE_PARALLEL_COMM = NCCLCommunicator(_PIPELINE_PARALLEL_RANK, args.cuda_id, group_size,
                                                   "pipeline_group_"+str(list_index))
        _PIPELINE_PARALLEL_COMM_B = NCCLCommunicator(_PIPELINE_PARALLEL_RANK, args.cuda_id, group_size,
                                                   "pipeline_group_b"+str(list_index))
        global _PIPELINE_PARALLEL_GATHER_COMM
        global _PIPELINE_PARALLEL_SCATTER_COMM
        global _PIPELINE_PARALLEL_GATHER_COMM_B
        global _PIPELINE_PARALLEL_SCATTER_COMM_B

        gathercom_index, gather_group_size = find_list(mapping_rank, gathers_comm)

        if gathercom_index is not None:
            #print("")
            for i in range(gather_group_size):
                if gathers_comm[gathercom_index][i] == mapping_rank:
                    _PIPELINE_GATHER_RANK = i
                    _GATHE_WORLD_SIZE =gather_group_size
            print("Gather_id:"+str(_PIPELINE_GATHER_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(gathers_comm[gathercom_index])+",Gather group size:"+str(gather_group_size))
            _PIPELINE_PARALLEL_GATHER_COMM = NCCLCommunicator(_PIPELINE_GATHER_RANK, args.cuda_id, gather_group_size,
                                                       "pipeline_gather_group_" + str(gathercom_index))
            _PIPELINE_PARALLEL_GATHER_COMM_B = NCCLCommunicator(_PIPELINE_GATHER_RANK, args.cuda_id, gather_group_size,
                                                       "pipeline_gather_group_b" + str(gathercom_index))
        scattercom_index, scatter_group_size = find_list(mapping_rank, scatters_comm)
        if scattercom_index is not None:
            for i in range(scatter_group_size):
                if scatters_comm[scattercom_index][i] == mapping_rank:
                    _PIPELINE_SCATTER_RANK = i
                    _SCATTER_WORLD_SIZE = scatter_group_size
            print("Scatther_id:"+str(_PIPELINE_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(scatters_comm[scattercom_index])+",Scatter group size:"+str(scatter_group_size))
            _PIPELINE_PARALLEL_SCATTER_COMM = NCCLCommunicator(_PIPELINE_SCATTER_RANK, args.cuda_id, scatter_group_size,
                                                       "pipeline_scatter_group_" + str(scattercom_index))
            _PIPELINE_PARALLEL_SCATTER_COMM_B = NCCLCommunicator(_PIPELINE_SCATTER_RANK, args.cuda_id, scatter_group_size,
                                                       "pipeline_scatter_group_b" + str(scattercom_index))
        
        if args.data_group_size != 1:
            list_index, data_group_size = find_list(mapping_rank, data_parallel_config)
            args.data_group_size = data_group_size
            data_para_list = data_parallel_config[list_index]
            _DATA_PARALLEL_WORLD_SIZE = args.data_group_size
            # [0,0] []
            #_DATA_PARALLEL_RANK = args.rank
            for i in range(data_group_size):
                if data_para_list[i] == mapping_rank:
                    _DATA_PARALLEL_RANK = i
            print("Data Parallel id:"+str(_DATA_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(data_para_list)+",DATA group size:"+str(data_group_size))
            _DATA_PARALLEL_COMM = NCCLCommunicator(_DATA_PARALLEL_RANK, args.cuda_id, args.data_group_size,
                                                   "data_group_" + str(list_index))
    else:
        print("Not supported yet")
        assert False


# from .nccl_backend import *

# _DATA_PARALLEL_COMM = None
# _DATA_PARALLEL_RANK = None
# _DATA_PARALLEL_WORLD_SIZE = None

# _PIPELINE_PARALLEL_COMM = None
# _PIPELINE_PARALLEL_RANK = None
# _PIPELINE_PARALLEL_WORLD_SIZE = None


# def get_data_parallel_comm() -> NCCLCommunicator:
#     assert _DATA_PARALLEL_COMM is not None
#     return _DATA_PARALLEL_COMM


# def get_data_parallel_rank() -> int:
#     assert _DATA_PARALLEL_RANK is not None
#     return _DATA_PARALLEL_RANK


# def get_data_parallel_world_size() -> int:
#     assert _DATA_PARALLEL_WORLD_SIZE is not None
#     return _DATA_PARALLEL_WORLD_SIZE


# def get_pipeline_parallel_comm() -> NCCLCommunicator:
#     assert _PIPELINE_PARALLEL_COMM is not None
#     return _PIPELINE_PARALLEL_COMM


# def get_pipeline_parallel_rank() -> int:
#     assert _PIPELINE_PARALLEL_RANK is not None
#     return _PIPELINE_PARALLEL_RANK


# def get_pipeline_parallel_world_size() -> int:
#     assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
#     return _PIPELINE_PARALLEL_WORLD_SIZE


# def init_communicators(args):
#     default_init(args)
#     assert args.world_size == args.data_group_size * args.pipeline_group_size
#     if args.world_size == args.data_group_size * args.pipeline_group_size:
#         #    We do the following hard code alignment of communication groups:
#         #    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
#         #    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
#         #    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
#         #    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
#         assert args.world_size == args.data_group_size * args.pipeline_group_size
#         global _DATA_PARALLEL_COMM
#         global _PIPELINE_PARALLEL_COMM
#         global _DATA_PARALLEL_RANK
#         global _PIPELINE_PARALLEL_RANK
#         global _DATA_PARALLEL_WORLD_SIZE
#         global _PIPELINE_PARALLEL_WORLD_SIZE
#         # We use pipeline parallel by default.
#         _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size
#         _PIPELINE_PARALLEL_RANK = args.rank % args.pipeline_group_size
#         _PIPELINE_PARALLEL_COMM = NCCLCommunicator(_PIPELINE_PARALLEL_RANK, args.cuda_id, args.pipeline_group_size,
#                                                    "pipeline_group_"+str(args.rank // args.pipeline_group_size))
#         if args.data_group_size != 1:
#             _DATA_PARALLEL_WORLD_SIZE = args.data_group_size
#             _DATA_PARALLEL_RANK = args.rank // args.pipeline_group_size
#             _DATA_PARALLEL_COMM = NCCLCommunicator(_DATA_PARALLEL_RANK, args.cuda_id, args.data_group_size,
#                                                    "data_group_"+str(args.rank % args.pipeline_group_size))
#     else:
#         print("Not supported yet")
#         assert False