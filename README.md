# MoDE_Calvin

Add paper link and more here

## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:mbreuss/MoDE_Diffusion_Policy.git
export mode_ROOT=$(pwd)/MoDE_Calvin

```
Install requirements
(Note we provided a changed verison of pyhash, given numerous problems when installing it manually)
```bash
cd $mode_ROOT
conda create -n mode_env python=3.8
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
cd transformer_blocks
pip install -e .
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

For replication of the orginial training results I recommend to use 4 GPUs with a batch_size of 128 and train them for 25 (35) epochs for ABC (ABCD).

#### (Optional) Preprocessing with CALVIN
Since MDT uses action chunking, it needs to load multiple (~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually ~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:
```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```
##### Params:
Run this command to see more detailed information:
```shell
python preprocess/extract_by_key.py -h
```

## Evaluation

Download the pretrained models from this link: [https://drive.google.com/drive/folders/17kuCgMi_GcYz1WVzumtIsKuvPYGvidxT?usp=sharing](https://drive.google.com/drive/folders/17kuCgMi_GcYz1WVzumtIsKuvPYGvidxT?usp=sharing).

Important params:
* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)
Optional params:
* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data
Thanks to @ygtxr1997 for debugging the GPU utilization and providing a merge request.


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


## Citation

If you found the code usefull, please cite our work:

```bibtex
@inproceedings{
    reuss2024multimodal,
    ???
}
```