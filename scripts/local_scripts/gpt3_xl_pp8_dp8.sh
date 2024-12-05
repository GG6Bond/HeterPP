cd /workspace/Hybrid_PTAFM_1F1B
# source activate pytorch_p38

ip=$1 # master-ip
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=allreduce
# Change the script here for different settings.
############################################################
ga_step=$5
num_layers=$6
batch_size=$7
############################################################
## start TExt
echo -en "\n\033[37;40;7m ---START--- [Rank-id:$rank cuda-$cuda_id] ./script/local_script/gpt3_xl_pp8_dp8.sh  \033[0m \n"
echo -en "  \033[36;40;7m [Rank-id:$rank cuda-$cuda_id]\033[0m: world_size" $world_size $timestamp $dp_mode "  master-ip  $ip  ga_step: " $ga_step "   num_layers: "$num_layers " num-iters:"10"  batch_size:"$batch_size  "\n"


let "global_batch_size = $ga_step*$batch_size*12"

# TODO: 
# DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 6 --data-group-size 2"
DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode 1f1b --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 2 --data-group-size 5"
# DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode dtfm_gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 2 --data-group-size 5"
# DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode dtfm_1f1b --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 3 --data-group-size 5"

echo $DIST_CONF
MODEL_CONF="--embedding-dim 2048 --num-heads 16 --num-layers $num_layers --batch-size $batch_size --micro-batch-size 1"

# DIST_CONF="--rank 0 --cuda-id 0 --world-size 1 --pipeline-group-size 1 --data-group-size 1  --pp-mode gpipe --dp-mode central_ps --gradient-accumulate-step 2 --embedding-dim 2048 --num-heads 16 --num-layers 3 --batch-size 64 --micro-batch-size 1"



# if [ "$world_size" -ne 64 ] ######!!!!!!? 原来是 "$world_size" -ne 64 
# then
#   echo "Not correct number of nodes"
#   exit 1
# fi

log_mode=$8
log_path="./logs/${timestamp}_gpt3_xl_pp8_dp8_l${num_layers}_b${global_batch_size}_rank${rank}_${log_mode}"

if [ $# -eq 9 ]
then
  case=$9
  export NCCL_SOCKET_IFNAME=eth0 ## 网卡设置
  export GLOO_SOCKET_IFNAME=eth0 ## 网卡设置

  # 尝试不设置延迟启动
  # sh ./scripts/tc_scripts/heterogeneous_setup_case"$case".sh
  
  # 不打印pythonlog
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "${log_path}_heter${case}.log"
  # python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF

fi

echo "Benchmark training is done."
echo -en "\033[37;40;7m ---END--- [Rank-id:$rank cuda-$cuda_id] ./script/local_script/gpt3_xl_pp8_dp8.sh  \033[0m \n"

