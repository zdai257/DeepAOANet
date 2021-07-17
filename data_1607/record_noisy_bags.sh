#!/bin/bash
python3 /home/zdai/repos/GPSLoRaRX/create_noisy.py &
sleep 1
END=5
for ((i=1;i<=END;i++)); do
    for bagfile in *.bag; do
        csvname="${bagfile%.*}"
        rosbag record -O noisy$i/${csvname}.bag -x /kerberos/iq_arr __name:=my_bag &
        sleep 2
        rosbag play ${csvname}.bag
        rosnode kill /my_bag
    done
done
