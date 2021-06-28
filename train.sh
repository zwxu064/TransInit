batch=16
num_GPU=4
slicloss=1
dataset="pascal"
enable_run_script=true
enable_resume_unary=true
backbone="resnet152"
optimizer="SGD"
lr_scheduler="fixed"
enable_ti=true
enable_coco_pretrained=true
ti_net_init="transparent"
data_root="datasets/PASCAL/train_val"
disable_logit_consistency=false

if ${enable_resume_unary}; then
    lr=1e-6
    sp_lr=1e-6
    epochs=20
elif [ ${dataset} == "pascal" ]; then
    lr=0.007
    sp_lr=0.007
    epochs=60
elif [ ${dataset} == "coco" ]; then
    lr=0.1
    sp_lr=0.01
    epochs=40
fi

if ${enable_ti}; then
    ti_lr=1e-7
else
    ti_lr=0
fi

if [ ${num_GPU} == 1 ]; then
    gpu_ids="0"
    batch=14
elif [ ${num_GPU} == 2 ]; then
    gpu_ids="0,1"
elif [ ${num_GPU} == 3 ]; then
    gpu_ids="0,1,2"
elif [ ${num_GPU} == 4 ]; then
    gpu_ids="0,1,2,3"
fi

script_dir="checkpoints/scripts/${dataset}"
log_dir="checkpoints/logs/${dataset}"

mkdir -p "${script_dir}"
mkdir -p "${log_dir}"

bash_name="${script_dir}/GPU${num_GPU}_epoch${epochs}_slic${slicloss}_bz${batch}"
bash_name+="_${optimizer}_${lr_scheduler}_lr${lr}_splr${sp_lr}_tilr${ti_lr}_${backbone}"
bash_name+="_${ti_net_init}"
log_name="${log_dir}/seg_GPU${num_GPU}_epoch${epochs}_slic${slicloss}_bz${batch}"
log_name+="_${optimizer}_${lr_scheduler}_lr${lr}_splr${sp_lr}_tilr${ti_lr}_${backbone}"
log_name+="_${ti_net_init}"

if ${enable_resume_unary}; then
    bash_name+="_finetune"
    log_name+="_finetune"
fi

if ${enable_coco_pretrained}; then
    bash_name+="_oncoco"
    log_name+="_oncoco"
fi

if ${disable_logit_consistency}; then
    bash_name+="_nologitconsist"
    log_name+="_nologitconsist"
fi

bash_name+=".sh"
log_name+=".txt"

# Write batch head
echo -e "#!/bin/bash -l

python3 main.py \\
--dataset=\"${dataset}\" \\
--backbone=\"${backbone}\" \\
--batch_size=${batch} \\
--val_batch_size=${batch} \\
--base_size=512 \\
--crop_size=512 \\
--data_root=\"${data_root}\" \\
--gpu_ids=\"${gpu_ids}\" \\
--epochs=${epochs} \\
--lr=${lr} \\
--sp_lr=${sp_lr} \\
--optimizer=\"${optimizer}\" \\
--lr_scheduler=\"${lr_scheduler}\" \\
--sp_resume=\"pretrained/sp_fcn/SpixelNet_bsd_ckpt.tar\" \\
--slic_loss=${slicloss} \\" >> ${bash_name}

if ${enable_resume_unary}; then
    if [ ${dataset} == "pascal" ]; then
        if [ ${backbone} == "resnet101" ]; then
            echo -e "--deeplab_resume=\"pretrained/resnet101/ckpt_50.pth.tar\" \\" >> ${bash_name}
        elif [ ${backbone} == "resnet152" ]; then
            if ${enable_coco_pretrained}; then
                echo -e "--deeplab_resume=\"pretrained/resnet152_pascaloncoco/ckpt_22.pth.tar\" \\" >> ${bash_name}
            else
                echo -e "--deeplab_resume=\"pretrained/resnet152/ckpt_53.pth.tar\" \\" >> ${bash_name}
            fi
        fi
    fi
else
    if ${enable_coco_pretrained}; then
        echo -e "--coco_resume=\"pretrained/resnet152_coco/ckpt_40.pth.tar\" \\" >> ${bash_name}
    fi
fi

if ${enable_ti}; then
    echo -e "--enable_ti \\
--ti_lr=${ti_lr} \\
--ti_net_init=\"${ti_net_init}\" \\" >> ${bash_name}
fi

if [ ${dataset} == "pascal" ]; then
    echo -e "--enable_save_val \\" >> ${bash_name}
fi

if ${disable_logit_consistency}; then
    echo -e "--disable_logit_consistency \\" >> ${bash_name}
fi

eval "chmod 755 ${bash_name}"

# Run
if ${enable_run_script}; then
    eval "${bash_name}"
fi