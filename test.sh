#!/bin/bash
cd diffusion_B

function modify_python_code_B1(){
    perl -pi -e 's/A/B/ if $. == 14' eval_diffusion.py
}

modify_python_code_B1

function modify_python_code_B2(){
    perl -pi -e 's/A/B/ if $. == 7' configs.yml
}

modify_python_code_B2

cd ..

cd diffusion_C

function modify_python_code_C1(){
    perl -pi -e 's/A/C/ if $. == 14' eval_diffusion.py
}

modify_python_code_C1

function modify_python_code_C2(){
    perl -pi -e 's/A/C/ if $. == 7' configs.yml
}

modify_python_code_C2

cd ..

start=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python diffusion_A/eval_diffusion.py & CUDA_VISIBLE_DEVICES=1 python diffusion_B/eval_diffusion.py & CUDA_VISIBLE_DEVICES=2 python diffusion_C/eval_diffusion.py

end=$(date +%s)

take=$((end - start))

echo Testing time taken to execute commands is ${take} seconds.
