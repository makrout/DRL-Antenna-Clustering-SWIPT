# DRL-Antenna-Clustering-SWIPT
##### Yasser Al-Eryani, Mohamed Akrout, Ekram Hossain
This repository contains the code of the paper: [Antenna Clustering for Simultaneous Wireless Information and Power Transfer in a MIMO Full-Duplex System: A Deep Reinforcement Learning-Based Design](https://arxiv.org/abs/2002.06193)

## Abstract
We propose a novel antenna clustering-based method for simultaneous wireless information and power transfer (SWIPT) in a multiple-input multiple-output (MIMO) full-duplex (FD) system. For a point-to-point communication set up, the proposed method enables a wireless device with multiple antennas to simultaneously transmit information and harvest energy using the same time-frequency resources. And the energy transmitting device with multiple antennas simultaneously receives information from the energy harvesting (EH) device.  This is achieved by clustering the  antennas into two MIMO subsystems: one for information transmission (IT) and another for EH. Furthermore, the self-interference (SI) signal at the EH device caused by the FD mode of operation is harvested by the device. For implementation-friendly antenna clustering and MIMO precoding, we propose two methods: (i) a sub-optimal method based on relaxation of objective function in a combinatorial optimization problem, and (ii) a hybrid deep reinforcement learning (DRL)-based method, specifically, a deep deterministic policy gradient (DDPG)-deep double Q-network  (DDQN) method. Finally, we study the performances of the two implementation methods and compare them with the conventional time switching-based simultaneous wireless information and power transfere (SWIPT) technique. Our findings show that the proposed MIMO clustering-based SWIPT method gives a significant improvement in spectral efficiency compared to the time switching-based SWIPT method. In particular, the DRL-based method provides the highest spectral efficiency. Furthermore, numerical results show that, for the considered system set up, the number of antennas in each device should exceed three to mitigate self-interference to an acceptable level.

Feedback welcome! The corresponding author is Ekram Hossain: ekram.hossain@umanitoba.ca

# Matlab implementation
The folder **suboptimal_antenna_clustering_matlab** contains the Matlab code of the suboptimal antenna clustering with relaxation-based MIMO precoding. It contains three files:
  - *main_antenna_splitting.m*: this is the main file for the suboptimal antenna clustering algorithm and the precoding through CVX toolbox. It calls the file Antenna_Splitting.m to return two MIMO subsystems; one for information transmission and another for energy harvesting,
  - *Antenna_Splitting.m*: this function accepts the CSI information of a MIMO system and returns two submatrices for information transmission and energy harvesting MIMO subsystems,
  - *main_time_switching.m*: this code finds the spectral efficiency of a conventional full-duplex energy harvesting MIMO system using the time switching SWIPT scheme.  

# DRL implementation
Our DRL implementation was tested on Ubuntu 16.04 and 18.04. It requires `tensorflow-gpu 1.14` and `keras 2.1.6` as indicated in the `requirements.txt` file.
The folder **DRL_python** contains:
  - *DDQN/DDPG folders*: they contain the Keras/Tensorflow implementation of the DDQN and DDPG algorithms,
  - *utils folder*: it contains different utility functions related to the implementation of the DDQN/DDPG algorithms,
  - *Env folder*: it contains the implementation of the wireless environment,
  - *script folder*: it contains bash files to help the user to automatically set the virtual environment and install the required packages,
  - *requirements.txt*: it contains the list of the packages required to run the code,
  - *main.py*: this is the main file that initialises the RL agents and the environment, and launches the training or the inference step,
  - *run.sh*: this is a bash file that allows the user to run the main file with the input parameters.
  
## First-time setup

If this is your first time running the code, follow these steps:

1. Run `script/up` to create a virtual environment `.venv` with the required packages
2. Activate the virtual environment by running `source .venv/bin/activate`

## Running experiments
### Arguments

| Argument &nbsp; &nbsp; &nbsp; &nbsp; | Description | Values |
| :---         |     :---      |          :--- |
| --step    |  Model train/inference step | choose from ['train', 'inference'], 'train' (default)      |
| --M1         |     The number of transmitters      |  4 (default) |
| --M2         |     The number of receivers      |  4 (default) |
| --snr_M1     | SNR at M1   | 37 (default)      |
| --snr_M2     | SNR at M2   | 37 (default)      |
| --nb_episodes   | Number of training episodes     | 2500 (default)    |
| --episode_length     | Length of one episode   | 500 (default)      |
| --batch_size     | Batch size (experience replay)   | 64 (default)      |
| --consecutive_frames     | Number of consecutive frames  | 2 (default)      |
| --training_interval    | Network training frequency  | 30 (default)      |
| --out_dir    | the output directory  | 'experiments' (default)      |

### Call example:
```bash
# Run the DRL implementation with specific values of arguments
python main.py --step="train" --M1=4 --M2=4 --snr_M1=37 --snr_M2=37 --nb_episodes=2500 --episode_length=500 --batch_size=64 --consecutive_frames=2 --training_interval=30
```

## Credit
We used a similar algorithmic structure of Hugo Germain's [repository](https://github.com/germain-hug/Deep-RL-Keras) to implement DDPG and DDQN.

## Citing the paper (bib)

If you make use of out code, please make sure to cite our paper:
```
@article{al2020simultaneous,
  title={Simultaneous Energy Harvesting and Information Transmission in a MIMO Full-Duplex System: A Machine Learning-Based Design},
  author={Al-Eryani, Yasser and Akrout, Mohamed and Hossain, Ekram},
  journal={arXiv preprint arXiv:2002.06193},
  year={2020}
}
```
