#!/bin/bash
python3 /home/zdai/repos/GPSLoRaRX/inverse_array_idx_slicing.py &
sleep 1

for bagfile in *.bag; do
    csvname="${bagfile%.*}"
    rosbag record -O "inv_${csvname}.bag" -x /kerberos/iq_arr __name:=my_bag &
    sleep 2
    rosbag play ${csvname}.bag
    rosnode kill /my_bag
done
