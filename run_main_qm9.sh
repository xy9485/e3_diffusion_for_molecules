#!/bin/bash
cd /home/xue/repos/real_EDM/

./updateHostLaunchJSON.sh

mamba activate edm
# Run the script with the following command:
# python /home/xue/repos/EDM/e3_diffusion_for_molecules/main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --dataset qm9_second_half --save_model False

# python main_qm9.py --n_epochs 1500 --exp_name edm_qm9_noH --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 10 --ema_decay 0.9999 --dataset qm9 --remove_h --save_model False 

# eval_sample
python eval_sample.py --model_path outputs/edm_qm9 --n_samples 10_000

# To analyze the sample quality of molecules
# python eval_analyze.py --model_path outputs/edm_qm9 --n_samples 100

# Train conditional EDM
# python main_qm9.py --exp_name exp_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning alpha --dataset qm9_second_half

#conditional evaluation
# python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --property alpha --n_sweeps 10 --task qualitative