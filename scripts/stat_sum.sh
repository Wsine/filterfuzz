#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
  echo "Illegal number of parameters"
  echo "usage: $0 log_file.log [-e]"
  exit 1
fi

if [[ "$#" -eq 2 && "$2" == "-e" ]]; then
  mapfile -t img_ids < <(cat $1 | cut -d ' ' -f 5)
  echo "num of log lines: ${#img_ids[@]}"

  line_num=0
  epoch_split=()
  prev_img_id=0
  for img_id in "${img_ids[@]}"; do
    (( line_num+=1 ))
    if (( $img_id < $prev_img_id )); then
      epoch_split+=($line_num)
    fi
    prev_img_id=$img_id
  done
  echo "num of epoches: $((${#epoch_split[@]} + 1))"

  for i in {1,5,10,15}; do
    split="${epoch_split[$((i+1))]}"
    echo -n "num of epoch $i: "
    head -n$split $1 | awk -F ' - ' '{print $2}' | sort | uniq | wc -l
  done

  echo -n "num of epoch 20: "
fi

cat $1 | awk -F ' - ' '{print $2}' | sort | uniq | wc -l

