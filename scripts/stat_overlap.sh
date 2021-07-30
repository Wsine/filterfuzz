#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters"
  echo "usage: $0 log_file1.log log_file2.log"
  exit 1
fi

comm -12 <(cat $1 | awk -F ' - ' '{print $2}' | sort | uniq) \
         <(cat $2 | awk -F ' - ' '{print $2}' | sort | uniq) \
     | wc -l

