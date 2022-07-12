# structural-triangulation


The official implementation of Structural Triangulation method.

## Files

All source files are listed below:

```
root
  |- README.md
  |- requirements.txt
  |- config.py
  |- structural_triangulation.py
  |- test.py
  |- utils.py
  `- virtual_test.py
```

Functions of the source code are:

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

## Tests

`Pose3D_inference` in `structural_triangulation.py` is the key function which implements the main method in our work. This function takes one frame of 2D poses, along with camera matrices, bone lengths, etc., as input, and produces the optimal 3D pose of the current frame. It is the basis for all tests. Actually, Structural Triangulation is as simple as just a triangulation method, these test files are simply sample codes. You may test this method however you like, as long as proper variables are passed in.

For tests on Human3.6M Dataset, 2D estimation part of [this model](https://github.com/karfly/learnable-triangulation-pytorch) is used as the 2D backbone. Just prepare data as [the guide](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md) suggests and dump necessary info (details see comments in config.py) to a pickle file. Then run `test()` in `test.py` will do.

Tests on other datasets and other 2D backbones are simply done in the same manner. Just prepare necessary data and edit path and settings in `config.py`.

Virtual test needs only the 3D ground truth as 2D estimations are generated. Use ground truth data prepared following this [guide](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md) and run `virtual_test.py` then the result will be output to `vir_result` folder.