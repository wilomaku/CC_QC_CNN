# Automatic CNN-based model for quality control of corpus callosum segmentation in large-scale studies

## Reproducible paper

This Readme file holds instructions for reproducing the study **Automatic quality control on corpus callosum segmentation: Comparing deep and classical machine learning approaches**

![Alt text](images/Graphical_abstract.png?raw=true "Title")

## Environment, used libraries and dependencies

* Python 3.5.4
* Numpy 1.12.1
* Scipy 0.19.1
* Matplotlib 2.0.2
* Pytorch 1.5.0
* aux library (My library, avaliable on: https://github.com/wilomaku/CC_QC_CNN/tree/master/aux)

## Workflow

This framework receives the $T_1$-MRI image and the binary mask, in nifti format (.nii.gz or .nii), and returns a quality score ranging from 0% - for completely correct segmentation - to 100% - for completely incorrect segmentation. The model was trained in 549 corpus callosum segmentations and tested in 136 samples.

## Files structure

The structure of the tree folder is located in the root of the git repository **wilomaku/CC_QC_CNN**:

* aux: Directory with main (func.py) functions.
* funcs: Empty directory. Only used for compatibility purposes.
* images: Required images for notebook visualization. The user should not modify it.
* saves: The files labels.csv and data_split_80_20.txt were only used to train the model and are no longer necessary for the user.

These files are located in the root of the git repository **wilomaku/CC_seg_clas**:

* README.md: File with the repository instructions.
* main.ipynb: Jupyter notebook with the training and testing the CNN-based model in a step-by-step fashion.
* test.py: Script to test among your segmentations. Useful if you want to test on your own dataset.

## Instructions to use this repository:

Please pay attention to these instructions and follow carefully. Before proceeding, you must guarantee that your system is as similar as possible to the environment, used libraries and dependencies described before.

1. Move to your directory: cd <your_dir>
2. Clone the repository: git clone https://github.com/wilomaku/CC_QC_CNN.git
3. If you want to run/train/test any file on this framework, you first need to change the DIR_BAS and DIR_SAVE variables to your paths.

### Test script (You want to perform quality control on your own segmentation dataset)

4. Download the saved model, with the trained parameters, available in: https://drive.google.com/file/d/1aBYKNaKJqgb0Mfu40W0S0cJKi7UorDZ0/view?usp=sharing. Put the model in the 'saves' folder.
5. You need to have your test dataset. The model only works with the pair image+binary mask, both in nifti format (.nii or .nii.gz are the only extensions accepted). The images and masks must be in a folder (<your_test_dir>), having mask and image for each subject in its respective folder:

```markdown
- __<your_test_dir>__
  - __folder1__
    - [image1.nii]
    - [mask1.nii]
  - __folder2__
    - [image2.nii]
    - [mask2.nii]
  .
  .
  .
```

Notice the image must contains 'image' in the name of the file, and the mask must contains 'mask' in the name of the file. It is expected that the nifti mask files are in 2D (in sagittal view) or 3D (in which case, the first dimension refers to the sagittal view). Copy the test dataset into your directory: cp <your_dir>/<your_test_dir>

6. Run the test script providing the proper arguments: python test.py

7. After executed, the output file with the quality score will be available in the save directory.

### Train script (You want to train the framework using your dataset)

4. You need to have your dataset. The framework only works with binary nifti masks (.nii or .nii.gz are the only extensions accepted). The images and masks must be in a folder (<your_test_dir>), having mask and image for each subject in its respective folder:

```markdown
- __<your_test_dir>__
  - __folder1__
    - [image1.nii]
    - [mask1.nii]
  - __folder2__
    - [image2.nii]
    - [mask2.nii]
  .
  .
  .
```

It is expected that the nifti mask files are in 2D (in sagittal view) or 3D (in which case, the first dimension refers to the sagittal view). Copy the test dataset into your directory: cp <your_dir>/<your_test_dir>

5. You need to have the proper labels for training the model. The label files must be in a csv file named **labels.csv** with two columns: 'Subject', containing the path or a partial identifier of the name; and 'Label', containing the label associated to every Subject (0 for correct segmentation and 1 for incorrect segmentation).

6. Set the hyper-parameters according to your dataset. Run the notebook **main.ipynb** to make sure your configuration and outputs are working as expected before run the train script.

Questions? Suggestions? Please write to wjgarciah@unal.edu.co

MIT License Copyright (c) 2019 William Herrera
