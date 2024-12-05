source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  # ssh -i $pem_file root@"$ip" -p $port "bash -s" < ./local_scripts/foo_load_lib.sh &
  ssh root@"$ip" -p 222 "bash -s" < ./local_scripts/foo_load_lib.sh &
done
wait