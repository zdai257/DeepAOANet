#!/bin/bash
dir1=data_0107
dir2=data_0207
END=30
for ((i=30;i<=END;i++)); do
    cd /home/zdai/repos/GPSLoRaRX/${dir1}
    for bagfile1 in deg_*.bag; do
        csvname1="${bagfile1%.*}"
        degname1="${csvname1#*_}"
        cd /home/zdai/repos/GPSLoRaRX/${dir2}
        for bagfile2 in deg_*.bag; do
            csvname2="${bagfile2%.*}"
            degname2="${csvname2#*_}"
            if [[ "$degname1" != "$degname2" ]]; then
                echo $degname1
                echo $degname2
                python3 /home/zdai/repos/GPSLoRaRX/Multilabel_Create_Synthetic.py &
                cd /home/zdai/repos/GPSLoRaRX/data_multi-${END}
                rosbag record -O ${degname1}_multi-${degname2}.bag /kerberos/R_multi __name:=my_bag &
                sleep 2
                cd /home/zdai/repos/GPSLoRaRX/${dir1}
                rosbag play -r 1 ${csvname1}.bag /kerberos/iq_arr:=/IQ1 &
                cd /home/zdai/repos/GPSLoRaRX/${dir2}
                rosbag play -r 1 ${csvname2}.bag /kerberos/iq_arr:=/IQ2
                sleep 5
                rosnode kill -a
            fi
        done
    done
done
