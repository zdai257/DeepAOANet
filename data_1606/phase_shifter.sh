#!/bin/bash
theta=-50
thetaname="m50"
m2="m52"
m4="m54"
p2="m48"
p4="m46"

python3 /home/zdai/repos/GPSLoRaRX/create_phaser.py --theta $theta --inverse True &
sleep 1

for bagfile in deg_${thetaname}.bag; do
    csvname="${bagfile%.*}"
    rosbag record -O inv_deg_${m2}.bag -e "(.*)_${m2}(.*)" __name:=my_bag &
    rosbag record -O inv_deg_${m4}.bag -e "(.*)_${m4}(.*)" __name:=my_bag &
    rosbag record -O inv_deg_${p2}.bag -e "(.*)_${p2}(.*)" __name:=my_bag &
    rosbag record -O inv_deg_${p4}.bag -e "(.*)_${p4}(.*)" __name:=my_bag &
    sleep 2
    rosbag play ${bagfile}
    rosnode kill -a
done
