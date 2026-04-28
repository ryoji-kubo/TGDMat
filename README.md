# TGDMat: Periodic Materials Generation using Text-Guided Joint Diffusion Model (ICLR 2025)

[![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://arxiv.org/pdf/2503.00522)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/kdmsit/TGDMat/)


Code for the ICLR 2025 paper [*"Periodic Materials Generation using Text-Guided Joint Diffusion Model"*](https://arxiv.org/pdf/2503.00522), 
by [Kishalay Das](https://kdmsit.github.io/), 
Subhojyoti Khastagir, 
[Pawan Goyal](https://cse.iitkgp.ac.in/~pawang/), 
Seung-Cheol Lee, 
[Satadeep Bhattacharjee](linkedin.com/in/satadeep-bhattacharjee-545567114/),
and [Niloy Ganguly](https://niloy-ganguly.github.io/).


TGDMat introduces a novel approach to generating 3D periodic materials using a text-guided diffusion framework:
- TGDMat is the first model to connect natural language understanding with the generation of 3D periodic materials.
- Simultaneously generates atomic coordinates, element types, and lattice parameters, while preserving essential periodic symmetry.
- Leverages rich, descriptive prompts to guide the creation process, enabling generation aligned with specific material properties and user intent.
- Outperforms existing state-of-the-art methods in accuracy and generalizability, with reduced training and inference costs.

![](TGDMat.png)

## Installation
The list of dependencies is provided in the `requirements.txt` file, generated using `pipreqs`. Yiu can install through following commands:
```bash
pip install -r requirements.txt
```
However, there may be some ad-hoc dependencies that were not captured. 
If you encounter any missing packages, feel free to install them manually using `pip install`.

## Installation-Ryoji
```bash
conda create --name tgdmat python=3.10 -y
conda activate tgdmat
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter==2.1.2 torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install -r requirements.txt
# I have relaxed the numpy and scipy version specifications in requirements.txt for the installation to work. For more details see the commented out lines in requirements.txt
```

## Ryoji-Notes

Generation task (short prompts):

For editing text: the attributes are controleled inside `generate_task/model/data_utils.py`, `prepare_text()`. 


## Textual Dataset
Text-guided reverse diffusion remains unexplored in material design, partly due to the lack of textual data in benchmark databases. To address this, we propose two methods for generating material descriptions: 
- (1) Using Robocrystallographer for detailed structural texts, and 
- (2) Creating shorter, user-friendly prompts with basic material info like chemical formula, elements, crystal system, and space group.

We kept the textual data for Perov-5, Carbon-24 and MP-20 databases in `data_text/` directory.
## Usage

### Crystal Structure Prediction(CSP) Task
Move to 'csp_task' directory

#### Train TGDMat Model

```bash
    python -W ignore train.py --dataset <Dataset> --batch_size 512 --epochs 500 --prompt_type <long/short>

    # Leave-one-out strategy
  python -W ignore train.py \
  --dataset mp_20 \
  --prompt_type short \
  --short_prompt_training_scheme leave_one_out \
  --batch_size 512 \
  --epochs 500
```

 - Where <Dataset> is perov_5/carbon_24/mp_20
 - Model saved at  out/<Dataset>/<expt_date>/<expt_time>/   

##### Evaluate TGDMat Model for CSP Task with #sample(k) = 1
```bash
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path  <saved_model_path> --tasks csp --num_evals 1 --dataset <Dataset> --batch_size 1024 --timesteps 1000 --prompt_type <long/short>  
python compute_metrics.py --root_path gen/perov_5/ --tasks recon
```

##### Evaluate TGDMat Model for CSP Task with #sample(k) = 20
```bash
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path  <saved_model_path> --tasks csp --num_evals 20 --dataset <Dataset> --batch_size 1024 --timesteps 1000 --prompt_type <long/short>  
python compute_metrics.py --root_path gen/perov_5/ --tasks recon --multi_eval
```

##### Reproduction
```bash
# tgdmat_full
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path checkpoints/mp20_default.pt  --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short
# metrics
python compute_metrics.py --root_path gen/mp_20/ --tasks recon

# mp_20 default with k = 20
CUDA_VISIBLE_DEVICES=1 python -W ignore evaluate.py --model_path 'gen_k@20/' --chkpt_path checkpoints/mp20_default.pt  --tasks csp --num_evals 20 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short

# omit
python -W ignore evaluate.py --model_path 'gen_omit_fenergy/' --chkpt_path checkpoints/mp20_default.pt  --tasks csp --num_evals 1 --dataset mp_20 --batch_size 1024 --timesteps 1000 --prompt_type short --short_prompt_eval_omit formation_energy_per_atom
```

| Model | Dataset | Short Prompt Training Scheme | Evaluation Scheme | # Samples | Match Rate | RMSE |
|---|---|---:|---:|---:|---:| ---:|
| tgdmat_full | mp_20 | full | full | 20 | 0.8004 |0.0782 |
| tgdmat_full | mp_20 | full | full | 1 | 0.5095 |0.0993 |
| tgdmat_full | mp_20 | full | omit_fenergy | 1 | 0.4575 |0.0994 |
| tgdmat_full | mp_20 | full | omit_band_gap | 1 | 0.4558 |0.1048 |
| tgdmat_full | mp_20 | full | omit_e_above_hull | 1 |0.4511|0.0992|
| tgdmat_full | mp_20 | full | omit_elements | 1 |0.4364|0.1035|
| tgdmat_full | mp_20 | full | omit_pretty_formula | 1 |0.4515|0.106|
| tgdmat_full | mp_20 | full | omit_spacegroup | 1 |0.4195 |0.1071 |
| tgdmat_full | mp_20 | full | omit_system | 1 |0.4134 |0.0983 |
| tgdmat_full | mp_20 | loo | full | 1 | 0.5021| 0.1079|
| tgdmat_full | mp_20 | loo | omit_fernergy | 1 |0.4954|0.1059|
| tgdmat_full | mp_20 | loo | omit_band_gap | 1 |0.4975|0.0995|
| tgdmat_full | mp_20 | loo | omit_e_above_hull | 1 |0.4994|0.104|
| tgdmat_full | mp_20 | loo | omit_elements | 1 |0.4948|0.1044|
| tgdmat_full | mp_20 | loo | omit_pretty_formula | 1 |0.5048|0.1032|
| tgdmat_full | mp_20 | loo | omit_spacegroup | 1 |0.4575|0.1057|
| tgdmat_full | mp_20 | loo | omit_system | 1 |0.4639|0.1033|

**Camera Ready Result**

| Model  | Omitted Feature | Match Rate | 
|---|---|---:|
| TGDMat  | None | 0.5095 |
| TGDMat  | fenergy | 0.4575 |
| TGDMat  | band_gap | 0.4558 |
| TGDMat  | e_above_hull | 0.4511 |
| TGDMat  | elements | 0.4364 |
| TGDMat  | pretty_formula | 0.4515 |
| TGDMat  | spacegroup | 0.4195 |
| TGDMat  | system | 0.4134 |
|---|---|---:|
| TGDMat_eval  | None | 0.5021 |
| TGDMat_eval  | fenergy | 0.4954 |
| TGDMat_eval  | band_gap | 0.4975 |
| TGDMat_eval  | e_above_hull | 0.4994|
| TGDMat_eval  | elements | 0.4948 |
| TGDMat_eval  | pretty_formula | 0.5048 |
| TGDMat_eval  | spacegroup | 0.4575|
| TGDMat_eval  | system | 0.4639 |

### Random Material Generation(Gen) Task
Move to 'generate_task' directory

##### Train TGDMat Model
```bash
python -W ignore train.py --dataset <Dataset> --batch_size 512 --epochs 500 --prompt_type <long/short>
```
 - Where <Dataset> is perov_5/carbon_24/mp_20
 - Model saved at  out/<Dataset>/<expt_date>/<expt_time>/

 To train it with leave-one-out strategy, do the following:
 ```bash
python -W ignore train.py \
  --dataset mp_20 \
  --prompt_type short \
  --short_prompt_training_scheme leave_one_out \
  --batch_size 512 \
  --epochs 500
 ```

##### Evaluate TGDMat Model for Material Generation Task
```bash
# Evaluation
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path  <saved_model_path> --tasks gen --dataset <Dataset> --batch_size 1024 --prompt_type <long/short>
# Metrics
python -W ignore compute_metrics.py --root_path gen/<Dataset>/ --tasks gen --gt_file <Test dtaset csv file path>

# tgdmat_full
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path checkpoints/mp20_default.pt --tasks gen --dataset mp_20 --batch_size 1024 --prompt_type short
# metrics
python -W ignore compute_metrics.py --root_path gen/mp_20/ --tasks gen --gt_file /home/ryoji/TGDMat/data_text/mp_20/test.csv

# apparently step_lr is wrong?
# tgdmat_full_step_lr_5e-6
python -W ignore evaluate.py --model_path 'gen_step_lr/' --chkpt_path checkpoints/mp20_default.pt --tasks gen --dataset mp_20 --batch_size 1024 --prompt_type short --step_lr 5e-6
# metrics
python -W ignore compute_metrics.py --root_path gen_step_lr/mp_20/ --tasks gen --gt_file /home/ryoji/TGDMat/data_text/mp_20/test.csv


# tgdmat_loo
CUDA_VISIBLE_DEVICES=1 python -W ignore evaluate.py --model_path 'gen_loo_full/' --chkpt_path checkpoints/mp20_loo.pt --tasks gen --dataset mp_20 --batch_size 1024 --prompt_type short
# metrics
CUDA_VISIBLE_DEVICES=1 python -W ignore compute_metrics.py --root_path gen_loo_full/mp_20/ --tasks gen --gt_file /home/ryoji/TGDMat/data_text/mp_20/test.csv

# tgdmat_full_omit_fenergy
CUDA_VISIBLE_DEVICES=2 python -W ignore evaluate.py --model_path 'gen_omit_formation_energy_per_atom/' --chkpt_path checkpoints/mp20_default.pt --tasks gen --dataset mp_20 --batch_size 1024 --prompt_type short --short_prompt_eval_omit formation_energy_per_atom
# metrics
CUDA_VISIBLE_DEVICES=2 python -W ignore compute_metrics.py --root_path gen_omit_formation_energy_per_atom/mp_20/ --tasks gen --gt_file /home/ryoji/TGDMat/data_text/mp_20/test.csv


# tgdmat_loo_omit_fenergy
CUDA_VISIBLE_DEVICES=3 python -W ignore evaluate.py --model_path 'gen_loo_omit_formation_energy_per_atom/' --chkpt_path checkpoints/mp20_loo.pt --tasks gen --dataset mp_20 --batch_size 1024 --prompt_type short --short_prompt_eval_omit formation_energy_per_atom
# metrics
CUDA_VISIBLE_DEVICES=3 python -W ignore compute_metrics.py --root_path gen_loo_omit_formation_energy_per_atom/mp_20/ --tasks gen --gt_file /home/ryoji/TGDMat/data_text/mp_20/test.csv

```
**Results**
| Model | Dataset | Short Prompt Training Scheme | Evaluation Scheme | Comp_Valid | Struct_Valid | wdist_density | wdist_num_elems | cov_recall | cov_precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tgdmat_full | mp_20 | full | full | 0.9891 |0.9998 |6.3507 |1.9102 |0.8609  |0.733 |
| tgdmat_full_step_lr_5e-6 | mp_20 | full|full |0.9898|1.0 |6.2765|1.9062| 0.8744| 0.7331|
| tgdmat_loo | mp_20 | loo | full |0.9916 |0.9999 |6.3322 |1.9122 |0.8731| 0.7332|
|tgdmat_full_omit_fenergy| mp_20| full| omit_fenergy |0.9915 |0.9996 |6.3834 | 1.9052| 0.8836| 0.7336|
|tgdmat_loo_omit_fenergy| mp_20 | loo| omit_fenergy |0.9919|0.9999|6.308|1.8982|0.8098|0.7322

**Camera-Ready Result**
| Model  | Comp_Valid | Struct_Valid | wdist_density | wdist_num_elems | cov_recall | cov_precision |
|---|---|---:|---:|---:|---:|---:|
| TGDMat (Reported) | 0.8660 |1.0000 |0.3296 |0.3337 |0.9979 |0.9988 |
| TGDMat (Reproduction) | 0.9891 |0.9998 |6.3507 |1.9102 |0.8609  |0.733 |




For any further query, feel free to contact [Kishalay Das](kishalaydas@kgpian.iitkgp.ac.in)

## How to cite

If you are using TGDMat or our Textuak Dataset, please cite our work as follow :

```
@article{das2025periodic,
  title={Periodic Materials Generation using Text-Guided Joint Diffusion Model},
  author={Das, Kishalay and Khastagir, Subhojyoti and Goyal, Pawan and Lee, Seung-Cheol and Bhattacharjee, Satadeep and Ganguly, Niloy},
  journal={arXiv preprint arXiv:2503.00522},
  year={2025}
}
```
