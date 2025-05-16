[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-diffusion-transformer-policies-with/zero-shot-generalization-on-calvin)](https://paperswithcode.com/sota/zero-shot-generalization-on-calvin?p=efficient-diffusion-transformer-policies-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-diffusion-transformer-policies-with/on-libero-10)](https://paperswithcode.com/sota/on-libero-10?p=efficient-diffusion-transformer-policies-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-diffusion-transformer-policies-with/on-libero-90)](https://paperswithcode.com/sota/on-libero-90?p=efficient-diffusion-transformer-policies-with)

# MoDE_Diffusion_Policy

[Paper](https://arxiv.org/pdf/2412.12953), [Project Page](https://mbreuss.github.io/MoDE_Diffusion_Policy/), [ICLR 25](https://openreview.net/pdf?id=nDmwloEl3N)


[Moritz Reuss](https://mbreuss.github.io/)<sup>1</sup>,
[Jyothish Pari](https://jyopari.github.io/aboutMe.html)<sup>2</sup>,
[Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/)<sup>2</sup>,
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab, KIT
<sup>2</sup>MIT CSAIL

## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:intuitive-robots/MoDE_Diffusion_Policy.git
export mode_ROOT=$(pwd)/MoDE_Diffusion_Policy

```
Install requirements
(Note we provided a changed verison of pyhash, given numerous problems when installing it manually)
```bash
cd $mode_ROOT
conda create -n mode_env python=3.9
conda activate mode_env
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
```
Next we can install the rest of the missing packages

```
pip install -r requirements.txt
```

---

## Download
### CALVIN Dataset

If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $mode_ROOT/dataset
sh download_data.sh D | ABCD
```

## Training
To train the mode model with the maximum amount of available GPUS, run:
```
python mode/training.py
```

For replication of the orginial training results I recommend to use 4 GPUs with a batch_size of 128 and train them for 15k steps for CALVIN and LIBERO results, when startin from pretrained weights.
See configs for details.

#### Preprocessing with CALVIN
Since MoDE uses action chunking, it needs to load multiple (~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually ~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:
```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```


```
python preprocess/extract_by_key.py -i /hkfs/work/workspace/scratch/ft4740-play3/data --in_task all
```

##### Params:
Run this command to see more detailed information:
```shell
python preprocess/extract_by_key.py -h
```


Important params:
* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)
Optional params:
* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data
Thanks to @ygtxr1997 for debugging the GPU utilization and providing a merge request.


After that modify the following line to add the storage path for your CALVIN dataset: [L9 in config_calvin.yaml](https://github.com/intuitive-robots/MoDE_Diffusion_Policy/blob/main/conf/config_calvin.yaml#L9)

## Evaluation

Download the pretrained models from Hugging Face: 
- [MoDE_CALVIN_ABC](https://huggingface.co/mbreuss/MoDE_CALVIN_ABC_1) 
- [MoDE_CALVIN_ABCD](https://huggingface.co/mbreuss/MoDE_CALVIN_ABCD)
- [MoDE_CALVIN_D](https://huggingface.co/mbreuss/MoDE_CALVIN_D)


## Results on Calvin ABC→D

| Method        | Active Params (Million) | PrT    | 1      | 2      | 3      | 4      | 5      | Avg. Len.        |
|---------------|-------------------------|--------|--------|--------|--------|--------|--------|-----------------|
| Diff-P-CNN    | 321                     | ×      | 63.5%  | 35.3%  | 19.4%  | 10.7%  | 6.4%   | 1.35±0.05        |
| Diff-P-T      | 194                     | ×      | 62.2%  | 30.9%  | 13.2%  | 5.0%   | 1.6%   | 1.13±0.02        |
| RoboFlamingo  | 1000                    | ✓      | 82.4%  | 61.9%  | 46.6%  | 33.1%  | 23.5%  | 2.47±0.00        |
| SuSIE         | 860+                    | ✓      | 87.0%  | 69.0%  | 49.0%  | 38.0%  | 26.0%  | 2.69±0.00        |
| GR-1          | 130                     | ✓      | 85.4%  | 71.2%  | 59.6%  | 49.7%  | 40.1%  | 3.06±0.00        |
| **MoDE**      | 307                     | ×      | 91.5%  | 79.2%  | 67.3%  | 55.8%  | 45.3%  | 3.39±0.03        |
| **MoDE**      | 436                     | ✓      | **96.2%** | **88.9%** | **81.1%** | **71.8%** | **63.5%** | **4.01±0.04** |

## Results on Calvin ABCD→D

| Method        | Active Params (Million) | PrT    | 1      | 2      | 3      | 4      | 5      | Avg. Len.        |
|---------------|-------------------------|--------|--------|--------|--------|--------|--------|-----------------|
| Diff-P-CNN    | 321                     | ×      | 86.3%  | 72.7%  | 60.1%  | 51.2%  | 41.7%  | 3.16±0.06        |
| Diff-P-T      | 194                     | ×      | 78.3%  | 53.9%  | 33.8%  | 20.4%  | 11.3%  | 1.98±0.09        |
| RoboFlamingo  | 1000                    | ✓      | 96.4%  | 89.6%  | 82.4%  | 74.0%  | 66.0%  | 4.09±0.00        |
| GR-1          | 130                     | ✓      | 94.9%  | 89.6%  | 84.4%  | 78.9%  | 73.1%  | 4.21±0.00        |
| **MoDE**      | 277                     | ×      | 96.6%  | 90.6%  | 86.6%  | 80.9%  | 75.5%  | 4.30±0.02        |
| **MoDE**      | 436                     | ✓      | **97.1%** | **92.5%** | **87.9%** | **83.5%** | **77.9%** | **4.39±0.04** |

We also provide the pretrained checkpoint after pretraining MoDE for 300k steps on a small OXE subset:

- [MoDE_pret](https://huggingface.co/mbreuss/MoDE_Pretrained) 

We used the following split for training:

| **Dataset** | **Weight** |
|-------------|------------|
| BC-Z | 0.258768 |
| LIBERO-10 | 0.043649 |
| BRIDGE | 0.188043 |
| CMU Play-Fusion | 0.101486 |
| Google Fractal | 0.162878 |
| DOBB-E | 0.245176 |
| **Total** | 1.000000 |

The model was pretrained for 300k steps with full pretraining details provided here; [MoDE Pretraining Report](https://api.wandb.ai/links/irl-masterthesis/ql9m7m5i).

---

## Acknowledgements

This work is only possible becauase of the code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)

#### BESO
Original: [https://github.com/intuitive-robots/beso](https://github.com/intuitive-robots/beso)
License: [MIT](https://github.com/intuitive-robots/beso/blob/main/LICENSE)

#### HULC
Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)
License: [MIT](https://github.com/lukashermann/hulc/blob/main/LICENSE)

#### MDT 

Original: [https://github.com/intuitive-robots/mdt_policy](https://github.com/intuitive-robots/mdt_policy)
License: [https://github.com/intuitive-robots/mdt_policy/blob/main/LICENSE](https://github.com/intuitive-robots/mdt_policy/blob/main/LICENSE) 

## Citation

If you found the code usefull, please cite our work:

```bibtex
@inproceedings{
    reuss2025efficient,
    title={Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers for Multitask Learning},
    author={Moritz Reuss and Jyothish Pari and Pulkit Agrawal and Rudolf Lioutikov},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=nDmwloEl3N}
}
```
