# structural-triangulation

The official implementation of ECCV 2022 paper *Structural Triangulation: A Closed-Form Solution to Constrained 3D Human Pose Estiation*.

## Files

All source files are listed below:

```
root
  |- README.md
  |- requirements.txt
  |- bl_estimate.py
  |- config.py
  |- structural_triangulation.py
  |- test.py
  |- utils.py
  `- virtual_test.py
```

Functions of the source code are:

* `bl_estimate.py`: The implementation of a simple way to estimate bone lengths with given frame indices: Taking the average over all symmetric bones.
* `config.py`: The configuration file, including the file paths and dataset info. 
* `structural_triangulation.py`: The main implementation file of Structural Triangulation. Inside there is a tree structure class (`DictTree`) and an estimation function (`Pose3D_estimate`).
* `test.py`: The test file for public datasets.
* `utils.py`: Implementations of some basic functions.
* `virtual_test.py`: The test file for virtual tests.


## Requirements

The solution is of closed form, so only some basic scientific calculation packages (`numpy`, `scipy`), and as visualization packages (`matplotlib`, `tqdm`) are needed. Note that Python 3.8+ is needed to make formatted strings work.

To install all requirements, simply

```shell
pip install -r requirements.txt
```

## How to Use

`Pose3D_inference(...)` in `structural_triangulation.py` is the key function which implements the main method in our work. This function takes one frame of 2D poses, along with camera matrices, bone lengths, etc., as input, and produces the optimal 3D pose of the current frame. Besides the closed-form Structural Triangulation combining with Step Constraint Method, implementation of Lagrangian Algorithm is also provided as a baseline. The methods are selected by a string parameter in `Pose3D_inference(...)`.

Actually, Structural Triangulation is as simple as just a triangulation method, these test files are more likely to be sample code than official ones, since it requires 2D estimations to be ready. You may test this method according to the following instruction, or however you like, as long as proper variables are passed in the functions.

For tests on Human3.6M Dataset, 2D estimation part of [this model](https://github.com/karfly/learnable-triangulation-pytorch) is used as the 2D backbone. Just prepare data as [the guide](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md) suggests and dump necessary info (details see comments in config.py) to a pickle file. Edit data path in `config.py` and then run `test()` in `test.py`.

Tests on other datasets and other 2D backbones are simply done in the same manner. Just prepare necessary data and edit path and settings in `config.py`.

Virtual test needs only the 3D ground truth as 2D estimations are generated. Use ground truth data prepared following this [guide](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md) and run `virtual_test.py` then the result will be output to `vir_result` folder.

## Citation
If you use our code, please cite with:
```
@inproceedings{Chen2022ECCV,
  title={Structural Triangulation\: A Closed-Form Solution to Constrained 3D Human Pose Estiation},
  author={Chen, Zhuo and Zhao, Xu and Wan, Xiaoyue},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
