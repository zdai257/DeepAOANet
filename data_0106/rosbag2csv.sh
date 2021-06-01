#!/bin/bash
for bagfile in *.bag; do
    csvname="${bagfile%.*}"
    rostopic echo -b "$bagfile" -p /kerberos/r > "${csvname}.csv"
done
