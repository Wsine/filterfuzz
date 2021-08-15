#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
  echo "Illegal number of parameters"
  echo "usage: $0 log_file.log [-e]"
  exit 1
fi

# https://zh.wikipedia.org/wiki/多样性指数

echo "Margalef's richness"
S=$(cat $1 | awk -F ' - ' '{print $3}' | sort | uniq | wc -l)
echo "S = $S"
n=$(cat $1 | awk -F ' - ' '{print $2}' | sort | uniq | wc -l)
D=$(bc -l <<< "($S - 1) / l($n)")
echo "$D"

echo "Shannon's diversity index"
p=(`cat $1 | awk -F ' - ' '{print $2" - "$3}' | sort | uniq | awk 'NF>1{print $NF}' | sort | uniq -c | awk '{print $1}'`)
sum_p=$(IFS=+; echo "$((${p[*]}))")
declare -a t=()
for i in "${p[@]}"; do
  t+=( "$(bc -l <<< "-($i / $sum_p) * l($i / $sum_p)")" )
done
H=$(IFS=+; bc <<< "${t[*]}")
echo "$H"

echo "Pielou's evenness index"
S=${#p[@]}
H_max=$(bc -l <<< "$H / l($S)")
echo "$H_max"

echo "Gini's index"
declare -a pi2=()
for i in "${p[@]}"; do
  pi2+=( "$(bc <<< "scale=10; ($i / $sum_p) * ($i / $sum_p)")" )
done
sum_pi2=$(IFS=+; bc <<< "${pi2[*]}")
gini=$(bc <<< "1 - $sum_pi2")
echo $gini
