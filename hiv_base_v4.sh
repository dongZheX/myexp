# /bin/sh
# Base Runs Graphormer in HIV Dataset

n_gpu=4
epoch=8
max_epoch=$((epoch + 1))
batch_size=64
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates/10))
RUNS=1
name="hiv_base_v4"
seeds=(1 2 3 4 5 6 7 8 9 10)
if [ ! -d "exp/logs/${name}" ]; then
    mkdir exp/logs/${name}
fi
if [ ! -d "exp/checkpoints_dir/${name}" ]; then
    mkdir exp/checkpoints_dir/${name}
fi
if [ ! -d "exp/results/${name}" ]; then
    mkdir exp/results/${name}
fi
for i in `seq 0 $((RUNS-1))`;
do
save_dir_root="exp/checkpoints_dir/${name}/RUN_${i}"
tensorboard_dir_root="exp/logs/${name}/RUN_${i}"
log_dir="exp/logs/${name}/RUN_${i}/metric_${i}.log"
result_dir="exp/results/${name}/result_${i}.log"
if [ ! -d "$tensorboard_dir_root" ]; then
    mkdir $tensorboard_dir_root
fi
if [ ! -d "$save_dir_root" ]; then
    mkdir $save_dir_root
fi
#TRAINNING
CUDA_VISIBLE_DEVICES=1,2,3,6 fairseq-train \
--user-dir graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name ogbg-molhiv \
--dataset-source ogb \
--task graph_prediction_with_flag \
--criterion binary_logloss_with_flag \
--arch graphormer_base \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size $batch_size \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch $max_epoch \
--save-dir $save_dir_root \
--pretrained-model-name pcqm4mv1_graphormer_base \
--seed ${seeds[$i]} \
--flag-m 3 \
--flag-step-size 0.001 \
--flag-mag 0.001 \
--tensorboard-logdir $tensorboard_dir_root \
--log-format simple --log-interval 100 \
--log-file $log_dir \
# --no-epoch-checkpoints
# --no-last-checkpoints
# --keep-best-checkpoints
# --save-interval
#EVALUATE

CUDA_VISIBLE_DEVICES=3 python graphormer/evaluate/evaluate.py \
    --user-dir graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molhiv \
    --dataset-source ogb \
    --task graph_prediction \
    --arch graphormer_base \
    --num-classes 1 \
    --batch-size $batch_size \
    --save-dir $save_dir_root  \
    --metric auc \
    --seed ${seeds[$i]} \
    --sfilename $result_dir \
    --log-format simple   \

done