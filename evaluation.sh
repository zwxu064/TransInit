enable_run_script=true
dataset="pascal"
backbone="resnet152"
enable_vanilla_deeplab=false
enable_adjust_val=true
enable_save_all=false
enable_save_png=false
enable_coco_pretrained=true
enable_finetune=true
enable_ti=true
slicloss=1
data_root="datasets/PASCAL/train_val"
adjust_val_factor=16

if ${enable_adjust_val}; then
    val_bz=1
else
    val_bz=16
fi

if ${enable_vanilla_deeplab}; then
    slicloss=0
    adjust_val_factor=16
    enable_ti=false
    enable_finetune=false
fi

if [ ${dataset} == "coco" ]; then
    resume="pretrained/resnet152_coco/ckpt_40.pth.tar"
else
    if [ ${backbone} == "resnet101" ]; then
        if ${enable_vanilla_deeplab}; then
            resume="pretrained/resnet101/ckpt_50.pth.tar"
        else
            resume="pretrained/resnet101_slic/ckpt_41.pth.tar"
            slicloss=1
        fi
    else
        if ${enable_coco_pretrained}; then
            if ${enable_vanilla_deeplab}; then
                resume="pretrained/resnet152_pascaloncoco/ckpt_22.pth.tar"
            else
                slicloss=1
                if ${enable_finetune}; then
                    if ${enable_ti}; then
                        resume="pretrained/resnet152_pascaloncoco_slic_ti_finetune/ckpt_15.pth.tar"
                    fi
                else
                    resume="pretrained/resnet152_pascaloncoco_slic/ckpt_52.pth.tar"
                fi
            fi
        else
            if ${enable_vanilla_deeplab}; then
                resume="pretrained/resnet152/ckpt_53.pth.tar"
            else
                slicloss=1
                if ${enable_finetune}; then
                    if ${enable_ti}; then
                        resume="pretrained/resnet152_slic_ti_finetune/ckpt_11.pth.tar"
                    else
                        resume="pretrained/resnet152_slic_finetune/ckpt_8.pth.tar"
                    fi
                else
                    resume="pretrained/resnet152_slic/ckpt_59.pth.tar"
                fi
            fi
        fi
    fi
fi

script_dir="checkpoints/scripts/${dataset}/eval"
log_dir="checkpoints/logs/${dataset}/eval"

mkdir -p "${script_dir}"
mkdir -p "${log_dir}"

bash_name="${script_dir}/slic${slicloss}_${backbone}"
log_name="${log_dir}/slic${slicloss}_${backbone}"

if ${enable_vanilla_deeplab}; then
    bash_name+="_vanilla"
    log_name+="_vanilla"
fi

if ${enable_coco_pretrained}; then
    bash_name+="_oncoco"
    log_name+="_oncoco"
fi

if ${enable_finetune}; then
    bash_name+="_finetune"
    log_name+="_finetune"
fi

if ${enable_ti}; then
    bash_name+="_ti"
    log_name+="_ti"
fi

if ${enable_adjust_val}; then
    bash_name+="_adjustval"
    log_name+="_adjustval"
fi

bash_name+=".sh"
log_name+=".txt"

# Write batch head
echo -e "#!/bin/bash -l

python3 main.py \\
--dataset=\"${dataset}\" \\
--backbone=\"${backbone}\" \\
--val_batch_size=${val_bz} \\
--gpu_ids=\"0\" \\
--data_root=\"${data_root}\" \\
--sp_resume=\"pretrained/sp_fcn/SpixelNet_bsd_ckpt.tar\" \\
--enable_save_val \\
--enable_test \\
--adjust_val_factor=${adjust_val_factor} \\
--slic_loss=${slicloss} \\
--resume=\"${resume}\" \\" > ${bash_name}

if ${enable_adjust_val}; then
    echo -e "--enable_adjust_val \\" >> ${bash_name}
fi

if ${enable_save_all}; then
    echo -e "--enable_save_all \\" >> ${bash_name}
fi

if ${enable_vanilla_deeplab}; then
    echo -e "--enable_vanilla \\" >> ${bash_name}
fi

if ${enable_save_png}; then
    echo -e "--enable_save_png \\" >> ${bash_name}
fi

if ${enable_ti}; then
    echo -e "--enable_ti \\" >> ${bash_name}
fi

eval "chmod 755 ${bash_name}"

# Run
if ${enable_run_script}; then
    eval "./${bash_name}"
fi