<img align="right" height="200" src="https://s1.52poke.wiki/wiki/thumb/c/c5/161Sentret.png/300px-161Sentret.png">

# FilterFuzz

Project Code: Sentret

For more details, please refer to the following publication.

## Publication

Z. Wei and W. K. Chan, "Fuzzing Deep Learning Models against Natural Robustness with Filter Coverage‡," 2021 IEEE 21st International Conference on Software Quality, Reliability and Security (QRS), 2021, pp. 608-619, doi: 10.1109/QRS54544.2021.00071.

## Preparation

The dataset needs to be downloaded manually. Please refer to [data/README.md](data/README.md).

## Installation

The project is maintained with [Pipenv](https://pipenv.pypa.io/en/latest/). Please refer to the link for installing Pipenv.

The dependencies are very convenient to install by one command. The versions are same as proposed here.

```bash
pipenv sync
```

## How to run

The executions are well organized with the help of Pipenv.

```
~/workspace/filterfuzz{main} > pipenv scripts
Command         Script
--------------  -------------------------------------------------
evaluate        python src/eval.py -d gtsrb -m convstn -p none
fuzz            python src/fuzz.py -d gtsrb -m convstn -p negconv
stat_sum        ./scripts/stat_sum.sh
stat_diversity  ./scripts/stat_diversity.sh
```
