# beta3_IRT
Source code of the paper [$\beta^3$-IRT: A New Item Response Model and its Applications](https://arxiv.org/abs/1903.04016)

## Requirements
The source code was originally developed on:
1. Python 2.7.12
2. Tensorflow 1.2.0 
3. [Edward](https://github.com/blei-lab/edward) 1.3.4

It was also tested on:
1. Python 3.6.6
2. Tensorflow 1.10.0
3. [Edward](https://github.com/blei-lab/edward) 1.3.5

which requires manually fixing the compatible issues of Tensorflow (> 1.2.0) in Edward and install Edward from the source.


## Usage

There are two steps to run experiments:
1. Train classifiers and generate response data for the $\beta^3$ IRT model, for example, run the following command:    
    ```
    python gen_irt_data.py --dataset moons --data_size 400 --noise_fraction 0.2 --seed 42
    ```

2. The first step will automatically generate data files for the second step, and the file named with \"irt_data_*.csv\" is the input parameter of the command to run $\beta^3$ IRT model, i.e.:
    ```
    python betairt_test.py --IRT_dfile irt_data_moons_s400_f20_sd42_m12.csv --a_prior_mean 1. --a_prior_std 1.  
    ```

## Citing $\beta^3$-IRT
Biblatex entry:
```
@inproceedings{chen2019beta,
  title={$\beta^3$-IRT: A New Item Response Model and its Applications},
  author={Chen, Yu and Filho, Telmo Silva and Prud{\^e}ncio, Ricardo BC and Diethe, Tom and Flach, Peter},
  booktitle={Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS) },
  year={2019}
  }
```