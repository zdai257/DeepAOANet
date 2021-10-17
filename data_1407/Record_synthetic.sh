#!/bin/bash
# degree value
arg1=$1
# degree name
arg2=$2
END=81
for ((i=81;i<=END;i++)); do
    theta=$(($arg1))
    thetaname=$arg2
    python3 /home/zdai/repos/GPSLoRaRX/Create_Synthetic.py --theta $theta &
    for bagfile in deg_*.bag; do
        csvname="${bagfile%.*}"
        degname="${csvname#*_}"
        echo $theta
        echo $thetaname
        if [[ "$thetaname" == "$degname" ]]; then
            echo $degname
            rosbag record -O noisy$i/${csvname}.bag -x /kerberos/iq_arr __name:=my_bag &
            sleep 2
            rosbag play -r 0.5 ${csvname}.bag
            sleep 5
            rosnode kill -a
        fi
    done
done
