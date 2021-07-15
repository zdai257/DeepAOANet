#!/bin/bash
END=5
for ((i=1;i<=END;i++)); do
    for bagfile in "noisy$i"/*.bag; do
        echo $bagfile
        csvname="${bagfile%.*}"
        rostopic echo -b "$bagfile" -p /kerberos/R1e_4> "${csvname}_1e_4.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R5e_4> "${csvname}_5e_4.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R1e_5> "${csvname}_1e_5.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R5e_5> "${csvname}_5e_5.csv"
    done
done
