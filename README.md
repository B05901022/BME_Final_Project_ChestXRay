# BME_Final_Project_ChestXRay
Train a classifier for CheXpert dataset and visually explain the model.

## Requirements
### Train
* PyTorch
* Torchvision
* TensorboardX
* NumPy
* Matplotlib
* sklearn
* Pandas
* PIL
* RandAugment (from [Ildoonet 's GitHub](https://github.com/ildoonet/pytorch-randaugment))

### Visualize
* Captum

## 0. Dataset
[CheXpert](https://arxiv.org/abs/1901.07031) is one of the largest chest x-ray image dataset to date, which contains 224,316 chest radiographs of 65,240 patients. For gaining access to the CheXpert dataset, please send registration to [StanfordML 's website](https://stanfordmlgroup.github.io/competitions/chexpert/). The dataset we used here is the smaller dataset but might be improved if you use the larger version. After downloading the dataset, you can change the directory description in `config/` and `validation.py` to easily fit your own environment.

The file structure should be:
```
<Main Directory>  
└─── <Our Repo>
└─── CheXpert-v1.0/
└─── (opional) ChestX-ray14/
```
The working directory should be at our repository.
```
cd BME_Final_Project_ChestXRay/
```

## 1. Train a model from scratch
For training a new model, add a new config file in the `config/`. You can customize your training settings like the template file in `config/base_config.yaml`. PyTorch pretrained models are available except inception-v3. EfficientNet pretrained models from [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) are also available.
After adding a new config file, you can start training by
```
python3 main.py -c config/<your_config_file>
```
For monitoring training process, TensorboardX is available.
```
tensorboard --logdir log/<your_config_name>
```

## 2. Test
To generate predictions for [CheXpert Challenge](https://stanfordmlgroup.github.io/competitions/chexpert/)
```
python3 predict.py <input-data-csv> <output_prediction-csv>
``` 

## 3. Visualize
Our work supports the visualization from [pytorch/captum](https://github.com/pytorch/captum).To visualize your model of a validation image,
```
python3 visualization.py -i <which-validation-image> -m <model#> -u <if-using-cached-threshold> -s <search-space-for-new-threshold>
```

## 4. Pseudo-Label
Our work supports writing pseudo-label from [NIH Chest X-ray dataset](https://www.kaggle.com/nih-chest-xrays/data). Note that in current work, this method doesn't help.
```
python3 pseudo_labeller.py -o <output-pred-csv> -m <model#> -u <if-using-cached-threshold> -s <search-space-for-new-threshold> -r <label-resolution>
```