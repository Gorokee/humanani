## Pose-Guided Human Animation from a Single Image in the Wild 


[[Project Page](http://gvv.mpi-inf.mpg.de/projects/PoseGuidedHumanAnimation/)] 
[[Paper](https://arxiv.org/pdf/2012.03796.pdf)]
[[Video](https://www.youtube.com/watch?v=x7H0kKWzRFU)]




## Installation

To install the necessary dependencies run the following command:
```shell
    pip install -r requirements.txt
```
The code has been tested with Python 3.7, CUDA 10.1.

## Demo 

### Step1: Download Pretrained model

You will download the [pre-trained weight](https://www.dropbox.com/s/pmwa69n5jr0fy8k/checkpoint.zip?dl=0) 
and unzip the files under folder checkpoints
```bash
checkpoint
├── Garent
├── Rendernet
├── Silnet
```
### Step2: Input data processing

You will convert the single view body fitting results and fashion segmentation result into the format we use.
You will run matlab_code/convert_data.m where this requires the fashion segmentation results and single view 3D body model body fitting result.
For segmentation, we use [fashion segmentation method](https://github.com/Engineering-Course/CIHP_PGN), and for body model fitting we use [SMPL fitting method] (https://github.com/nkolot/SPIN).
where we save the vertices and camera translation. Please refer to the same data. 

### Step3: Complete UV map generation

To obtain a complete unified UV texture and garment labels from a single image, please run the following command:
```shell
   python UV_modeling.py
```
This will generate the complete UV maps and the intermediate results in the UV_model folder. 

### Step4: Inference

To synthesize the person image from different body pose, you will run the following command:
```shell
   python inference.py
```
This will generate the synthesized image in the output folder.


 
## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```bibtex
@inproceedings{yoon2020humanani,
  title={Pose-Guided Human Animation from a Single Image in the Wild},
  author={Jae Shin Yoon, Lingjie Liu, Vladislav Golyanik, Kripasindhu Sarkar, Hyun Soo Park, and Christian Theobalt},
  booktitle={CVPR},
  year={2021}
}
```


## Contact
The code of this repository was implemented by [Jae Shin Yoon](mailto:jsyoon@umn.edu).
