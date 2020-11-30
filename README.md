# DCGAN implementation using Pytorch

## prerequisite
Check the requirments.txt in the file and <br>

`pip install -r requirements.txt` <br>

### training 
Download the dataset you want to train and run: <br>
`python train.py --data_root <your data path>`

You can check the Setting.py to try different hyperparameters.

### visualization
When doing training, run <br>
`tensorboard --logdir=<your directory path>`

### Reference

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
