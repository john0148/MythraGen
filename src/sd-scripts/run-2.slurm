#!/bin/bash
#SBATCH --job-name=train-model # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=3 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node

# Activate conda environment for the first task
source /home/sv-lkhai/miniconda3/etc/profile.d/conda.sh

# Add the sd-scripts directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/sv-lkhai/sd-scripts

# Navigate to the project directory
cd /home/sv-lkhai/sd-scripts

# Prompt-1, Prompt-2
conda activate retrieval_env
# Run the first script
python PrepareForTraining.py "prompt-1.txt" "data"

# Deactivate the first conda environment
conda deactivate

# Activate conda environment for the second task
conda activate lora_env-2

# Run the second script
accelerate launch \
    --num_cpu_threads_per_process=4 \
    "/home/sv-lkhai/sd-scripts/train_network.py" \
    --config_file=/home/sv-lkhai/sd-scripts/data/lora_sdscipts_example_data/train_network_config.toml

python inferences.py "prompt-1.txt" "data"
python inferences_without_LoRA.py "prompt-1.txt" "data"

conda deactivate

conda activate retrieval_env

# Run the first script
python PrepareForTraining.py "prompt-2.txt" "data-2"

# Deactivate the first conda environment
conda deactivate

# Activate conda environment for the second task
conda activate lora_env-2

# Run the second script
accelerate launch \
    --num_cpu_threads_per_process=4 \
    "/home/sv-lkhai/sd-scripts/train_network.py" \
    --config_file=/home/sv-lkhai/sd-scripts/data-2/lora_sdscipts_example_data/train_network_config.toml

python inferences.py "prompt-2.txt" "data-2"
python inferences_without_LoRA.py "prompt-2.txt" "data-2"
conda deactivate

#Prompt-3, Prompt-4
# conda activate retrieval_env
# # Run the first script
# python PrepareForTraining.py "prompt-3.txt" "data-3"

# # Deactivate the first conda environment
# conda deactivate

# # Activate conda environment for the second task
# conda activate lora_env-2

# # Run the second script
# accelerate launch \
#     --num_cpu_threads_per_process=4 \
#     "/home/sv-lkhai/sd-scripts/train_network.py" \
#     --config_file=/home/sv-lkhai/sd-scripts/data-3/lora_sdscipts_example_data/train_network_config.toml

# python inferences.py "prompt-3.txt" "data-3"
# python inferences_without_LoRA.py "prompt-3.txt" "data-3"

# conda deactivate

# conda activate retrieval_env

# # Run the first script
# python PrepareForTraining.py "prompt-4.txt" "data-4"

# # Deactivate the first conda environment
# conda deactivate

# # Activate conda environment for the second task
# conda activate lora_env-2

# # Run the second script
# accelerate launch \
#     --num_cpu_threads_per_process=4 \
#     "/home/sv-lkhai/sd-scripts/train_network.py" \
#     --config_file=/home/sv-lkhai/sd-scripts/data-4/lora_sdscipts_example_data/train_network_config.toml

# python inferences.py "prompt-4.txt" "data-4"
# python inferences_without_LoRA.py "prompt-4.txt" "data-4"
# conda deactivate

#Prompt-5, Prompt-6
conda activate retrieval_env

# Run the first script
python PrepareForTraining.py "prompt-5.txt" "data-5"

# Deactivate the first conda environment
conda deactivate

# Activate conda environment for the second task
conda activate lora_env-2

# Run the second script
accelerate launch \
    --num_cpu_threads_per_process=4 \
    "/home/sv-lkhai/sd-scripts/train_network.py" \
    --config_file=/home/sv-lkhai/sd-scripts/data-5/lora_sdscipts_example_data/train_network_config.toml

python inferences.py "prompt-5.txt" "data-5"
python inferences_without_LoRA.py "prompt-5.txt" "data-5"
conda deactivate

conda activate retrieval_env

# Run the first script
python PrepareForTraining.py "prompt-6.txt" "data-6"

# Deactivate the first conda environment
conda deactivate

# Activate conda environment for the second task
conda activate lora_env-2

# Run the second script
accelerate launch \
    --num_cpu_threads_per_process=4 \
    "/home/sv-lkhai/sd-scripts/train_network.py" \
    --config_file=/home/sv-lkhai/sd-scripts/data-6/lora_sdscipts_example_data/train_network_config.toml

python inferences.py "prompt-6.txt" "data-6"
python inferences_without_LoRA.py "prompt-6.txt" "data-6"
conda deactivate



