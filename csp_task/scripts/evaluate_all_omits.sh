checkpoint_path=checkpoints/mp20_loo.pt 

python -W ignore evaluate.py --model_path 'gen_loo_omit_fenergy/' --chkpt_path $checkpoint_path  --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit formation_energy_per_atom
python -W ignore evaluate.py --model_path 'gen_loo_omit_band_gap/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit band_gap
python -W ignore evaluate.py --model_path 'gen_loo_omit_e_above_hull/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit e_above_hull
python -W ignore evaluate.py --model_path 'gen_loo_omit_pretty_formula/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit pretty_formula
python -W ignore evaluate.py --model_path 'gen_loo_omit_elements/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit elements
python -W ignore evaluate.py --model_path 'gen_loo_omit_spacegroup/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit spacegroup
python -W ignore evaluate.py --model_path 'gen_loo_omit_system/' --chkpt_path $checkpoint_path --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit system
