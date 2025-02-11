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