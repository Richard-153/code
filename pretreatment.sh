#!/bin/bash
cp -r diffusion_A/ diffusion_B

cd diffusion_B/models/

function modify_python_code_B(){
    perl -pi -e 's/from models.unet_A import DiffusionUNet/from models.unet_B import DiffusionUNet/g' ddm.py
}

modify_python_code_B

cd ..

function modify_python_code_B1(){
    perl -pi -e 's/A/B/ if $. == 5' configs.yml
}

modify_python_code_B1

function modify_python_code_B2(){
    perl -pi -e 's/A/B/ if $. == 6' configs.yml
}

modify_python_code_B2

function modify_python_code_B3(){
    perl -pi -e 's/A/B/ if $. == 15' train_diffusion.py
}

modify_python_code_B3

function modify_python_code_B4(){
    perl -pi -e 's/A/B/ if $. == 37' configs.yml
}

modify_python_code_B4

function modify_python_code_B5(){
    perl -pi -e 's/A/B/ if $. == 8' configs.yml
}

modify_python_code_B5

cd /root/workspace/cgh_workspace/four/

cp -r diffusion_A/ diffusion_C

cd diffusion_C/models/

function modify_python_code_C(){
    perl -pi -e 's/from models.unet_A import DiffusionUNet/from models.unet_C import DiffusionUNet/g' ddm.py
}

modify_python_code_C

cd ..

function modify_python_code_C1(){
    perl -pi -e 's/A/C/ if $. == 5' configs.yml
}

modify_python_code_C1

function modify_python_code_C2(){
    perl -pi -e 's/A/C/ if $. == 6' configs.yml
}

modify_python_code_C2

function modify_python_code_C3(){
    perl -pi -e 's/A/C/ if $. == 15' train_diffusion.py
}

modify_python_code_C3

function modify_python_code_C4(){
    perl -pi -e 's/A/C/ if $. == 37' configs.yml
}

modify_python_code_C4

function modify_python_code_C5(){
    perl -pi -e 's/A/C/ if $. == 8' configs.yml
}

modify_python_code_C5

cd ..

start=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python diffusion_A/train_diffusion.py > diffusion_A/logs/a.log & CUDA_VISIBLE_DEVICES=1 python diffusion_B/train_diffusion.py > diffusion_B/logs/b.log & CUDA_VISIBLE_DEVICES=2 python diffusion_C/train_diffusion.py > diffusion_C/logs/c.log

end=$(date +%s)

take=$((end - start))

echo Training time taken to execute commands is ${take} seconds.
