# A framework for quality control of corpus callosum segmentation in large-scale studies

## Reproducible paper

This Readme file holds instructions for reproducing the study **A framework for quality control of corpus callosum segmentation in large-scale studies**

![Alt text](images/Graphical_abstract.png?raw=true "Title")

## Environment, used libraries and dependencies

* Python 3.5.4
* Numpy 1.12.1
* Scipy 0.19.1
* Matplotlib 2.0.2
* Scikit-learn 0.19.0
* aux library (My library, avaliable on: https://github.com/wilomaku/CC_seg_clas/tree/master/aux)

## Workflow

This framework receives a binary mask, in nifti format (.nii.gz or .nii), and returns a quality score ranging from 0% - for completely correct segmentation - to 100% - for completely incorrect segmentation. The model was trained in 481 corpus callosum segmentations and tested in 207 samples.

![Alt text](images/Framework_quality.png?raw=true "Title")

## Files structure

The structure of the tree folder is located in the root of the git repository **wilomaku/CC_seg_clas**:

* aux: Directory with main (func.py) and auxiliar (aux_fnc.py) functions. The default configuration (**default_config.py**) for training and testing the framework is available too. The file labels.csv was only used to train the model and is no longer necessary for the user.
* funcs: Empty directory. Only used for compatibility purposes.
* images: Required images for notebook visualization. The user should not modify it.
* saves: Saved models used to test the framework. **arr_models_ind.joblib** contains the saved models for the individual classifiers; **ensemble_model.joblib** is the ensemble final model; and **sign_refs.joblib** has the partial extracted signatures.

These files are located in the root of the git repository **wilomaku/CC_seg_clas**:

* README.md: File with the repository instructions.
* main.ipynb: Jupyter notebook with the reproducible paper in a step-by-step fashion.
* main.py: Script to train and test the quality control framework. Useful if you have your own dataset to train.
* test.py: Script to test among your segmentations. Useful if you want to test on your own dataset.

## Instructions to use this repository:

Please pay attention to these instructions and follow carefully. Before proceeding, you must guarantee that your system is as similar as possible to the environment, used libraries and dependencies described before. If this is not possible, you can choose to use Docker (instructions below).

1. Move to your directory: cd <your_dir>
2. Clone the repository: git clone https://github.com/wilomaku/CC_seg_clas.git
3. If you want to run/train/test any file on this framework, you first need to change the DIR_BAS and DIR_SAVE variables to your paths in **default_config.py**.

### Test script (You want to perform quality control on your own segmentation dataset)

4. You need to have your test dataset. The framework only works with binary nifti masks (.nii or .nii.gz are the only extensions accepted). Your masks must be in a folder (<your_test_dir>), either directly in the root of <your_test_dir>:

```markdown
- __<your_test_dir>__
  - [mask1.nii]
  - [mask2.nii.gz]
  .
  .
  .
```

or every mask in its respective folder:
```markdown
- __<your_test_dir>__
  - __folder1__
    - [mask1.nii]
  - __folder2__
    - [mask2.nii]
  .
  .
  .
```

It is expected that the nifti mask files are in 2D (in sagittal view) or 3D (in which case, the first dimension refers to the sagittal view). Copy the test dataset into your directory: cp <your_dir>/<your_test_dir>

5. Run the test script providing the proper arguments: python test.py <dir_in> <pattern> <msp> <-opt_th>

dir_in: Databse input directory.
-pattern[Optional]: Input a string pattern that is present in all the nifti file names. If no particular pattern is present in your name files, do not pass anything.
-msp[Optional]: Slice to be selected on sagittal plane. If 3D masks are used, you must pass a slice with a valid mask. Please note that the by-default value is 90. If 2D masks are used, do not pass anything.
-opt_th[Optional]: Decision threshold to separate classes. If this value is not passed, the optimal threshold is used instead.

Examples: 
* python test.py /home/jovyan/work/dataset/ (Example with 2D masks with no particular pattern in file names)
* python test.py /home/jovyan/work/dataset/ -pattern mask -opt_th 0.5 (Example with 2D masks with 'mask' string present in file names to be evaluated. The decision threshold applied is 0.5)
* python test.py /home/jovyan/work/dataset/ -msp 100 -opt_th 0.5 (Example with 3D masks. The 100th sagittal slice is selected. The decision threshold applied is 0.5)

6. After executed, the output file with the quality score will be available in the save directory (the save directory path (DIR_SAVE) can be changed in **default_config.py**).

### Train script (You want to train the framework using your dataset)

4. You need to have your dataset. The framework only works with binary nifti masks (.nii or .nii.gz are the only extensions accepted). Your masks must be in a folder (<your_test_dir>) either directly in the root of <your_test_dir>:

```markdown
- __<your_test_dir>__
  - [mask1.nii]
  - [mask2.nii.gz]
  .
  .
  .
```

or every mask in its respective folder:
```markdown
- __<your_test_dir>__
  - __folder1__
    - [mask1.nii]
  - __folder2__
    - [mask2.nii]
  .
  .
  .
```

It is expected that the nifti mask files are in 2D (in sagittal view) or 3D (in which case, the first dimension refers to the sagittal view). Copy the test dataset into your directory: cp <your_dir>/<your_test_dir>

5. You need to have the proper labels for training the model. The label files must be in a csv file named **labels.csv** with two columns: 'Subject', containing the path or a partial identifier of the name; and 'Label', containing the label associated to every Subject (0 for correct segmentation and 1 for incorrect segmentation).

6. Set the hyper-parameters according to your dataset. I recommend you run the notebook **main.ipynb** to make sure your configuration and outputs are working as expected before run the train script. This notebook works in the same way as the train script.

7. Run the train script: python main.py. The script will save the trained models in the save directory (the save directory path (DIR_SAVE) can be changed in **default_config.py**).

### Instructions to execute on Docker image in either, test or train mode:

Because the model is fully dependant on the Scikit-learn version, I used a Docker image to guarantee reproducibility from now on. The Docker image fulfills all the software requirements and is only necessary to provide the cloned repository with the scripts. I used a public image in a Docker Hub with the required configuration to execute the scripts, including Nibabel to deal with nifti files.

4. Install Docker on your machine: https://docs.docker.com/install/

5. Download Docker image: docker pull miykael/nipype_level0 (https://hub.docker.com/r/miykael/nipype_level0)

6. Run Docker on Jupyter mode: docker run -p 8889:8888 -v ~/Documents/:/home/jovyan/work -it miykael/nipype_level0 (Notice that port 8888 is being mapped to port 8889 when opening Jupyter)

7. Run Docker on terminal mode: docker run -v ~/Documents/:/home/jovyan/work -it miykael/nipype_level0 /bin/bash

8. When in the Docker prompt, you can proceed with the Instructions to either **Test script** or **Train script** as explained previously.

## Original publication

We are pleased if you use our work. Kindly cite our paper:

Herrera, W. G., Pereira, M., Bento, M., Lapa, A. T., Appenzeller, S., & Rittner, L. (2020). A framework for quality control of corpus callosum segmentation in large-scale studies. Journal of Neuroscience Methods, 108593. (https://doi.org/10.1016/j.jneumeth.2020.108593).

Questions? Suggestions? Please write to wjgarciah@unal.edu.co

MIT License Copyright (c) 2019 William Herrera
