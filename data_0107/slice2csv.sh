#!/bin/bash
END=7
for bagfile in inv_*.bag; do
  echo "$bagfile"
  if [[ $bagfile =~ ^inv_* ]] ; then
    csvname0="${bagfile%.*}"
    csvname="${csvname0#*_}"
    #echo "$csvname"
    for ((i=0;i<=END;i++)); do
      echo "${csvname}-${i}.csv"
      rostopic echo -b "$bagfile" -p /kerberos/r > "${csvname}-${i}.csv"
    done
  fi
done
