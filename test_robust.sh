base_dir="./test_output/output_robust"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test_robust.py \
    --model BCAIML \
    --world_size 1 \
    --test_data_path "./data/datasets/CASIA1.0" \
    --checkpoint_path "./checkpoint/" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --seed 42 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log