# 将 GPT 扩展到任意模态

## Install

### 1.安装 Torch Deepspeed

``` bash
# Cuda 11.1.1
$ pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# deepspeed
$ pip install deepspeed
```

### 2.安装 Apex

``` bash
# 配置特定版本
$ git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
$ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### 3.安装 Mujoco

Download file

``` bash
$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
$ mkdir -p ~/.mujoco/mujoco210
$ tar -xzvf ./mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Add this to `.bashrc`:

``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Avoid error compiling Cython file:

``` bash
$ apt-get install libghc-x11-dev libglew-dev patchelf
$ pip install Cython==3.0.0a10
```

### 4.安装仿真环境

#### MetaWorld

```
pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
```

### 5.实验设置

#### MDP: 

$
s_{1} \stackrel{a_{1}}{\rightarrow} s_{2}, r_{1} \stackrel{a_{2}}{\rightarrow} s_{3}, r_{2}, ..., s_{t} \stackrel{a_{t}}{\rightarrow} s_{t+1}, r_{t}
$

#### Training object:

$s_{t}, a_{t} \rightarrow s_{t+1}$ 

$s_{t}, s_{t+1} \rightarrow a_{t}$ 

$s_{t}, s_{t+1} \rightarrow r_{t}$ 

+

$s_{t} \rightarrow a_{t}$ 

$s_{t} \rightarrow r_{t}$ 

#### Comparison

GPT-Based         | Repres-Based              
---               | ---        
Pretrain          | Pretrain
Sequence Modeling | $s_{t}, a_{t} \rightarrow s_{t+1}$ 
Reverse Modeling  | $s_{t}, s_{t+1} \rightarrow a_{t}$ 
Reverse Modeling  | $s_{t}, s_{t+1} \rightarrow r_{t}$ 
\                 | Fine-turn
Sequence Modeling | $s_{t} \rightarrow a_{t}$ 
Sequence Modeling | $s_{t} \rightarrow r_{t}$

#### Ablate exam:

GPT-Based          | Object
---                | ---        
Base               | $s_{t} \rightarrow a_{t}, \ s_{t+1} \rightarrow a_{t+1}$
Sequence Modeling  | $s_{t} \rightarrow a_{t} \rightarrow s_{t+1} \rightarrow a_{t+1} \rightarrow s_{t+2}$ 
Reverse Modeling   | $s_{t+1} \rightarrow a_{t+1} \rightarrow s_{t} \rightarrow a_{t}$ 
Bilateral Modeling | \


Repres-Based       | Object
---                | ---        
Base               | 
State Modeling     | $s_{t}, a_{t} \rightarrow s_{t+1}$ 
Action Modeling    | $s_{t}, s_{t+1} \rightarrow a_{t}$ 
Reward Modeling    | $s_{t}, s_{t+1} \rightarrow r_{t}$ 


> GPT-3 的尺度

parameters  | $n_{layers}$ | $d_{model}$ | $d_{head}$
---         | ---          | ---         | ---
125M        | 12           | 768         | 64
350M        | 24           | 1024        | 64
760M        | 24           | 1536        | 96
1.3B        | 24           | 2048        | 128


End-to-end unified pre-training and fine-tuning with