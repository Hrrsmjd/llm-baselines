# 1) Print your current directory
pwd

# 2) Make a directory (if it doesn't already exist)
mkdir my_huggingface_home

# 3) Export HF_HOME to that path (using 'pwd')
export HF_HOME="$(pwd)/my_huggingface_home"

---

Installed Jupyter and Matplotlib


---
Code to run:

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:4" --activation "xrelu2" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "xrelu2"


OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:1" --activation "xrelu3" --activation_p 0.001 --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "xrelu3"

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:2" --activation "mrelu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu"

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:4" --activation "mgelu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mgelu" --activation_a 1.0 --activation_p 1e-7 --activation_s 0.0 --activation_b 0.0

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:6" --activation "msilu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "msilu" --activation_a 1.0 --activation_p 1e-7 --activation_s 0.0 --activation_b 0.0

---

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:5" --activation "m2relu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "m2relu" --activation_a_n -0.1 --activation_a_p 1.0 --activation_p_p 1.0 --activation_s 0.0 --activation_b 0.0

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:6" --activation "m3relu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "m3relu" --activation_a_n 1e-7 --activation_a_p 1.0 --activation_p_n 1e-7 --activation_p_p 1.0 --activation_s 0.0 --activation_b 0.0

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:6" --activation "m3relu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "m3relu" --activation_a_n 1e-7 --activation_a_p 1.0 --activation_p_n 1e-7 --activation_p_p 1.0 --activation_s 0.0 --activation_b 0.0

---
## check parameters

0 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power False --activation_shift_h False --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

1 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:1" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power False --activation_shift_h False --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

2 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:2" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power True --activation_shift_h False --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

3 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:5" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power False --activation_shift_h True --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

4 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:6" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power False --activation_shift_h False --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

12 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power True --activation_shift_h False --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

13 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power False --activation_shift_h True --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

14 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power False --activation_shift_h False --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

23 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power True --activation_shift_h True --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

24 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power True --activation_shift_h False --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

34 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power False --activation_shift_h True --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

123 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:5" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power True --activation_shift_h True --activation_shift_v False --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

124 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:6" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power True --activation_shift_h False --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

134 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:0" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power False --activation_shift_h True --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

234 Done 
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:1" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale False --activation_power True --activation_shift_h True --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0

1234 Done
OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:2" --activation "mrelu_selective" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "mrelu_selective" --activation_scale True --activation_power True --activation_shift_h True --activation_shift_v True --activation_alpha 1.0 --activation_p 1.0 --activation_a 0.0 --activation_k 0.0