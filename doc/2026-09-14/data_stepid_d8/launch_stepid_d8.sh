#!/bin/bash
# 启动臂① @ depth-8: 守卫 + 训练后处理 + 训练 (全部 setsid nohup 脱离会话)
export PATH=/root/miniconda3/bin:$PATH
mkdir -p /root/train_logs
cd /root
setsid nohup bash /root/guard_stepid_d8.sh            >/root/train_logs/stepid_d8_guard.log   2>&1 &
sleep 1
setsid nohup bash /root/post_stepid_d8_slice05.sh     >/root/train_logs/stepid_d8_post.log    2>&1 &
sleep 1
setsid nohup bash /root/run_stepid_d8_slice05.sh      >/root/train_logs/stepid_d8_wrapper.log 2>&1 &
sleep 10
echo "== launched $(date '+%F %T') =="
ps -eo pid,ppid,etime,cmd | grep -E "stepid_d8|accelerate|train_v2" | grep -v grep
