#!/bin/bash
for bagfile in inv_*.bag; do
  echo "$bagfile"
  if [[ $bagfile =~ ^inv_* ]] ; then
    csvname="${bagfile%.*}"
    echo "$csvname"
    #rostopic echo -b "$bagfile" -p /kerberos/r > "${csvname}.csv"
    #rostopic echo -b "$bagfile" -p /kerberos/doa_results > "music_${csvname}.csv"
  fi
done
