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

OMP_NUM_THREADS=1 python ./src/main.py --config_format base --device "cuda:5" --activation "m3relu" --compile --wandb --wandb_project "llm-baselines" --wandb_run_prefix "m3relu" --activation_a_n 1e-7 --activation_a_p 1.0 --activation_p_n 1e-7 --activation_p_p 1.0 --activation_s 0.0 --activation_b 0.0