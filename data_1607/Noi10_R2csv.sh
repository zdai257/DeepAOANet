#!/bin/bash
END=14
for ((i=10;i<=END;i++)); do
    for bagfile in "noisy$i"/*.bag; do
        echo $bagfile
        csvname="${bagfile%.*}"
        rostopic echo -b "$bagfile" -p /kerberos/R1e_3> "${csvname}_1e_3.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R2e_3> "${csvname}_2e_3.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R3e_3> "${csvname}_3e_3.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R4e_3> "${csvname}_4e_3.csv"
        rostopic echo -b "$bagfile" -p /kerberos/R5e_3> "${csvname}_5e_3.csv"
    done
done
