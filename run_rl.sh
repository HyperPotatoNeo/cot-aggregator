#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --account=m5017
#SBATCH --constraint=gpu&hbm80g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --qos=regular


module load cuda/12.4 cudnn gcc mpich
source $SCRATCH/.virtualenvs/verl/bin/activate

export HF_HOME=$SCRATCH/.cache/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#if [ $# -lt 1 ]; then
#  echo "Usage: $0 <dataset_name>"
#  exit 1
#fi
#DATASET=$1

DATA_DIR="/pscratch/sd/m/mokshjn/eval/tower_of_hanoi"
TRAIN_FILE="${DATA_DIR}/agg_0.parquet"
VAL_FILE="/pscratch/sd/m/mokshjn/eval/games/eval.parquet"
OUTPUT_DIR="$DATA_DIR"


#export HF_HUB_OFFLINE=1

set -x

srun python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=256 \
    data.max_prompt_length=33792 \
    data.max_response_length=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=41984 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='cot-aggregator' \
    trainer.experiment_name='qwen3_4b_countdown_agg_1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000 \
    trainer.total_epochs=100 $@