# public IPs of AWS instances.



# =============================================== stage节点数（12 4 4 12 4）
# ips=(
# "192.168.100.6"
# "192.168.100.8"
# "192.168.0.6"
# "192.168.0.9"
# "192.168.0.2"
# "192.168.100.4"
# "192.168.100.7"
# "192.168.100.5"
# "192.168.100.12"
# "192.168.100.10"
# "192.168.100.13"
# "192.168.100.11"
# "192.168.100.14"
# "192.168.0.4"
# "192.168.0.5"
# )
# (6 3 1 2 8 2 1 2 1 2 1 2 1 2 2)




# ips=(
#     192.168.100.6
#     192.168.100.8
#     192.168.0.4
#     192.168.0.5
#     192.168.0.6
#     192.168.0.9
#     192.168.0.2
#     192.168.100.5
#     192.168.100.12
#     192.168.100.10
#     192.168.100.13
#     192.168.100.11
#     192.168.100.14
#     192.168.100.4
#     192.168.100.7
# )




# AAAAAAA =============每个stage节点数（3 1 1 3 1）
# ips=(
# "192.168.100.6"
# "192.168.0.4"
# "192.168.100.11"
# "192.168.100.14"
# "192.168.0.5"
# )


# 每个stage节点数（1 1 1 3 3）
# ips=(
# "192.168.0.4"
# "192.168.0.5"
# "192.168.100.11"
# "192.168.100.14"
# "192.168.100.6"
# )

# # 每个stage节点数（1 1 1 3 1）
# ips=(
# "192.168.0.4"
# "192.168.0.5"
# "192.168.100.11"
# "192.168.100.14"
# "192.168.0.2"
# )

# 消融实验 gather 和 scatter(9个节点)
# ips=(
# "192.168.100.6"
# "192.168.0.4"
# "192.168.0.5"
# "192.168.100.8"
# "192.168.100.11"
# )

# 消融实验 gather 和 scatter(全部节点)
# ips=(
# "192.168.100.6"
# "192.168.0.9"
# "192.168.0.2"
# "192.168.100.4"
# "192.168.100.7"
# "192.168.100.5"
# "192.168.100.12"
# "192.168.100.9"
# "192.168.100.13"
# "192.168.100.10"
# "192.168.100.14"
# "192.168.0.4"
# "192.168.0.5"
# "192.168.100.8"
# "192.168.100.11"
# "192.168.0.6"
# "192.168.0.7"
# )

# 所有T4
# ips=(
# "192.168.100.6"
# "192.168.0.9"
# "192.168.100.4"
# "192.168.100.7"
# "192.168.100.5"
# "192.168.100.12"
# "192.168.100.9"
# "192.168.100.13"
# "192.168.100.10"
# "192.168.100.14"
# "192.168.100.8"
# "192.168.100.11"
# )

# 10 A100
ips=(
    192.168.0.2
    192.168.0.4
)

port="222"
# master_ip="192.168.100.6"
master_ip="192.168.0.2"