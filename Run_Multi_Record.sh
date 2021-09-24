#!/bin/bash
END=70
for ((i=-70;i<=END;i=i+10)); do
    for ((j=-70;j<=END;j=j+10)); do
        iname=$i
        jname=$j
        if [[ $i == -* ]]; then
            degname="${i:1}"
            iname="m${degname}"
        fi
        if [[ $j == -* ]]; then
            degname="${j:1}"
            jname="m${degname}"
        fi
        #echo $i
        #echo $iname
        #echo $j
        #echo $jname
        bash Multilabel_Record_synthetic.sh $i $iname $j $jname
    done
done


