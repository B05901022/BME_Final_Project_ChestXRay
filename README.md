# BME_Final_Project_ChestXRay
Train a classifier for CheXpert dataset and visually explain the model.

## Requirements
* PyTorch
* Torchvision
* TensorboardX
* NumPy
* Matplotlib
* sklearn
* Pandas
* PIL

## 0. Dataset
[CheXpert](https://arxiv.org/abs/1901.07031) is one of the largest chest x-ray image dataset to date, which contains 224316 chest radiographs of 65240 patients.
For gaining access to the CheXpert dataset, please send registration to [StanfordML 's website](https://stanfordmlgroup.github.io/competitions/chexpert/).
The dataset we used here is the smaller dataset which is 11 GB, but might be improved if you use the larger version. 
After downloading the dataset, you can change the directory description in `config/` and `validation.py` to easily fit your own environment.

## 1. Train a model from scratch
For training a new model, add a new config file in the `config/`. You can customize your training settings like the template file in `config/base_config.yaml`.
PyTorch pretrained models are available except inception-v3. EfficientNet pretrained models from [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) are also available.
After adding a new config file, you can start training by
```
python3 main.py -c config/<your_config_file>
```
For monitoring training process, TensorboardX is available.
```
tensorboardX --logdir log/<your_config_name>
```

## 2. Testing
For generating predictions for the [CheXpert Challenge](https://stanfordmlgroup.github.io/competitions/chexpert/)
```
python3 validation.py <input-data-csv> <output_prediction-csv>
``` 