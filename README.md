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

### Random Material Generation(Gen) Task
Move to 'generate_task' directory

##### Train TGDMat Model
```bash
python -W ignore train.py --dataset <Dataset> --batch_size 512 --epochs 500 --prompt_type <long/short>
```
 - Where <Dataset> is perov_5/carbon_24/mp_20
 - Model saved at  out/<Dataset>/<expt_date>/<expt_time>/

##### Evaluate TGDMat Model for Material Generation Task
```bash
python -W ignore evaluate.py --model_path 'gen/' --chkpt_path  <saved_model_path> --tasks gen --dataset <Dataset> --batch_size 1024 --prompt_type <long/short>
python -W ignore compute_metrics.py --root_path gen/<Dataset>/ --tasks gen --gt_file <Test dtaset csv file path>
```


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
