source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh root@"$ip" -p 222 "bash -s" < ./local_scripts/local_clear_logs.sh &
done
wait