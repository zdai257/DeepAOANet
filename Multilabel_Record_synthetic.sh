#!/bin/bash
arg1=$1
arg2=$2
arg3=$3
arg4=$4
# degree value
theta1=$(($arg1))
theta2=$(($arg3))
# degree name
thetaname1=$arg2
thetaname2=$arg4
dir1=data_0207
dir2=data_1607
END=71
for ((i=END;i<=END;i++)); do
    cd /home/zdai/repos/GPSLoRaRX/${dir1}
    for bagfile1 in deg_*.bag; do
        csvname1="${bagfile1%.*}"
        degname1="${csvname1#*_}"
        cd /home/zdai/repos/GPSLoRaRX/${dir2}
        for bagfile2 in deg_*.bag; do
            csvname2="${bagfile2%.*}"
            degname2="${csvname2#*_}"
            if [ "$thetaname1" == "$degname1" ] && [ "$thetaname2" == "$degname2" ]; then
                echo $theta1
                echo $theta2
                python3 /home/zdai/repos/GPSLoRaRX/Multilabel_Create_Synthetic.py --theta1 $theta1 --theta2 $theta2 &
                cd /home/zdai/repos/GPSLoRaRX/data_multi-${END}
                rosbag record -O ${degname1}_multi-${degname2}.bag -e "/kerberos/R_(.*)" __name:=my_bag &
                sleep 2
                cd /home/zdai/repos/GPSLoRaRX/${dir1}
                rosbag play -r 0.5 ${csvname1}.bag /kerberos/iq_arr:=/IQ1 &
                cd /home/zdai/repos/GPSLoRaRX/${dir2}
                rosbag play -r 0.6 ${csvname2}.bag /kerberos/iq_arr:=/IQ2
                sleep 6
                rosnode kill -a
            fi
        done
    done
done
