#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "usage: $0 log_file.log"
  exit 1
fi

cat $1 | awk -F ' - ' '{print $2}' \
       | sort | uniq \
       | awk '{arr[$NF]++} END { for (k in arr) { print k ": " arr[k] } }'

