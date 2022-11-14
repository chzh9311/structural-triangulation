# structural-triangulation

The official implementation of ECCV 2022 paper [*Structural Triangulation: A Closed-Form Solution to Constrained 3D Human Pose Estiation*](https://doi.org/10.1007/978-3-031-20065-6_40).

## Files

All source files are listed below:

```
root
  |- README.md
  |- requirements.txt
  |- get_bone_lengths.py
  |- config.py
  |- structural_triangulation.py
  |- test.py
  |- utils.py
  |- virtual_test.py
  `- configs
        |- h36m_config.yaml
        `- virtual_config.yaml
```

Functions of the source code are:

* `get_bone_lengths.py`: The implementation of a simple way to estimate bone lengths with given frame indices: Taking the average over all symmetric bones.
* `config.py`: The interface to get configurations
* `structural_triangulation.py`: The main implementation file of Structural Triangulation. Inside there is a tree structure class (`DictTree`) and an estimation function (`Pose3D_estimate`).
* `test.py`: The test file for public datasets.
* `utils.py`: Implementations of some basic functions.
* `virtual_test.py`: The test file for virtual tests.
* Files under `configs` dir are configurations files used in experiments. You may modify specific terms according to your purpose.

## Requirements

The solution is of closed form, so only some basic scientific calculation packages (`numpy`, `scipy`), and as visualization packages (`matplotlib`, `tqdm`) are needed. Note that Python 3.8+ is needed to make formatted strings work.

To install all requirements, simply

```shell
pip install -r requirements.txt
```

## How to Use

`Pose3D_inference(...)` in `structural_triangulation.py` is the key function which implements the main method in our work. This function takes one frame of 2D poses, along with camera matrices, bone lengths, etc., as input, and produces the optimal 3D pose of the current frame. Besides the closed-form Structural Triangulation combining with Step Constraint Method, implementation of Lagrangian Algorithm is also provided as a baseline. The methods are selected by a string parameter in `Pose3D_inference(...)`.

Actually, Structural Triangulation is as simple as just a triangulation method, these test files are more likely to be sample code than official ones, since it requires 2D estimations to be ready. You may test this method however you like, as long as proper variables are passed in the functions.

If you're willing to use our code to reproduce the results in the paper, here are the instructions:

## Test on Public Dataset (Human3.6M)

### Data Preparation

For tests on Human3.6M Dataset, pre-processing and 2D backbone are provided by [this model](https://github.com/karfly/learnable-triangulation-pytorch). We made some modifications to dump 2D estimations and ground truth labels to a pickle file, you may download it from [here](https://drive.google.com/file/d/1wOMCBcOzypoGr-e05fjxSPGIF_bdEEu5/view?usp=sharing).

After that, make a directory named `data` and place the file in it, so that your local directory looks like this:

```
root
  |- data
  |    `- detected_data.pkl
  ...
```

Then, run `get_bone_lengths.py` to get bone lengths.

```shell
python get_bone_lengths.py
```

You will see npy files generated under `data/bone_lengths/h36m` dir:

```
root
  |- data
  |    |- detected_data.pkl
  |    `- bone_lengths/h36m
  |          |- S9_bl_estimated.npy
  ...        ...
```

Here, `*_estimated.npy` files contain results estimated from linear triangulation result of T-poses; `*_gt.npy` files contain that from ground truth. 

### Running test

With data ready, running test is very simple:

```shell
python test.py
```

The result will be printed on screen once the test is finished. You can modify configurations in `configs/h36m_config.yaml`.

## Test on Synthesized Dataset

Virtual test needs only the 3D ground truth as 2D estimations are generated. Just prepare data according to the previous section. Then run

```shell
python virtual_test.py
```

You will see results in `csv` format under `vir_result` folder. To specify camera numbers and 2D estimation errors, modify `configs/virtual_config.yaml`.

## ToDos:

Implement the functions to

* dump experimental results to local storage instead of just printing on screen;

* specify parameters in command arguments;

* process data in batches using GPU.

## Citation

If you use our code, please cite with:

```latex
@inproceedings{Chen2022ECCV,
  title={Structural Triangulation: A Closed-Form Solution to Constrained 3D Human Pose Estiation},
  author={Chen, Zhuo and Zhao, Xu and Wan, Xiaoyue},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
