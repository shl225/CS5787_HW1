CS5787 HW1 Deep Learning with Prof. Hadar Elor - Sean Hardesty Lewis (shl225)
# Implementing and Training the Lenet5 network over the FashionMNIST dataset
## Overview
This project implements several variants of the LeNet-5 architecture (originally described in LeCun et al., 1998) for classifying images from the FashionMNIST dataset. The variants include:

* Baseline LeNet-5 (No regularization)
* LeNet-5 with Dropout
* LeNet-5 with L2 Regularization
* LeNet-5 with Batch Normalization
* Optimized LeNet-5 with L2 Regularization

The goal is to compare the performance of these regularization techniques and achieve at least 88% accuracy on the test set.

*(The reason for the optimized L2 Regularization model with tuned hyperparameters is due to the original L2 Regularization model not scoring over 88% on our training or test.)*

## Architecture Modifications
The LeNet-5 architecture was originally designed for the MNIST dataset. Primary changes to adapt this architecture to FashionMNIST are:

* Input dimensions remain the same (28x28 grayscale images).
* Modified activation functions to use Tanh for feature extraction layers.
* Added variants with Dropout, L2 Regularization (weight decay), and Batch Normalization (per Prof. Elor's HW1 reqs).
### Specific Modifications
* **Dropout:** Introduced a Dropout layer (with a dropout rate of 0.5) in the fully connected part of the network to mitigate overfitting.
* **Batch Normalization:** Added Batch Normalization after convolutional layers to stabilize learning and accelerate training.
* **L2 Regularization:** Applied weight decay during optimization to penalize large weights improving our generalization.

## Training Settings and Hyperparameters
#### Hyperparameters:
* **Batch Size:** 64 (selected based on memory constraints and performance trade-offs).
* **Learning Rate:** 0.001, a common default that worked well without causing the training to become unstable.
* **Optimizer:** Adam for all models (Prof. Elor mentioned in lecture on 08/10/24 that only using Adam is okay for this).
* **Epochs:** Baseline, Dropout, Batch Norm, and initial L2 models were trained for 12 epochs. The optimized L2 model was trained for 20 epochs to achieve improved convergence.
* **Weight Decay:** Set to 0.01 for all models, reduced to 0.001 for the optimized model to balance underfitting and overfitting.
#### Train/Validation Split:
* **Split Ratio:** 90% training, 10% validation. Referenced [this SO post](https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio).
* **Method:** Used PyTorch's `random_split` to ensure a fair and random distribution of data into training and validation sets.
## Training Instructions
To train the models with my hyperparameters, use the following commands:

### Baseline model
```
train_acc_baseline, val_acc_baseline = train_model(
    model=model_baseline, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    epochs=12, 
    optimizer=Adam, 
    accuracy=Accuracy(task='multiclass', num_classes=10)
)
```
### Dropout model
```
train_acc_dropout, val_acc_dropout = train_model(
    model=model_dropout, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    epochs=12, 
    optimizer=Adam, 
    accuracy=Accuracy(task='multiclass', num_classes=10)
)
```
### L2 Regularization model
```
train_acc_l2, val_acc_l2 = train_model(
    model=model_l2, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    epochs=12, 
    optimizer=Adam, 
    accuracy=Accuracy(task='multiclass', num_classes=10),
    weight_decay=0.01 
)
```
### Optimized L2 Regularization model
```
train_acc_l2_optimized, val_acc_l2_optimized = train_model(
    model=model_l2, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    epochs=20,  # Increased epochs for better convergence
    optimizer=Adam,
    accuracy=Accuracy(task='multiclass', num_classes=10),
    weight_decay=0.001,  # Reduced weight decay for better generalization
)
```
### Batch Normalization model
```
train_acc_batch_norm, val_acc_batch_norm = train_model(
    model=model_batch_norm, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    epochs=12, 
    optimizer=Adam, 
    accuracy=Accuracy(task='multiclass', num_classes=10)
)
```

## Saving the Weights
To save the weights of the trained models, use the following commands:

### Baseline Model
```
torch.save(model_baseline.state_dict(), MODEL_SAVE_PATHS['baseline'])
```
### Dropout Model
```
torch.save(model_dropout.state_dict(), MODEL_SAVE_PATHS['dropout'])
```
### L2 Regularization Model
```
torch.save(model_l2.state_dict(), MODEL_SAVE_PATHS['l2_regularization'])
```
### Optimized L2 Regularization Model
```
torch.save(model_l2.state_dict(), MODEL_PATH / "lenet5v1_l2_optimized.pth")
```
### Batch Normalization Model
```
torch.save(model_batch_norm.state_dict(), MODEL_SAVE_PATHS['batch_norm'])
```

## Testing with Saved Weights
To test the models with saved weights, use the following commands:

### Baseline Model
```
model_baseline_loaded = LeNet5V1()
model_baseline_loaded.load_state_dict(torch.load(MODEL_SAVE_PATHS['baseline']))
```

### Dropout Model
```
model_dropout_loaded = LeNet5V1(use_dropout=True)
model_dropout_loaded.load_state_dict(torch.load(MODEL_SAVE_PATHS['dropout']))
```
### L2 Regularization Model
```
model_l2_loaded = LeNet5V1()
model_l2_loaded.load_state_dict(torch.load(MODEL_SAVE_PATHS['l2_regularization']))
```
### Optimized L2 Regularization Model
```
model_l2_optimized_loaded = LeNet5V1()
model_l2_optimized_loaded.load_state_dict(torch.load(MODEL_PATH / "lenet5v1_l2_optimized.pth"))
```
### Batch Normalization Model
```
model_batch_norm_loaded = LeNet5V1(use_batch_norm=True)
model_batch_norm_loaded.load_state_dict(torch.load(MODEL_SAVE_PATHS['batch_norm']))
```

## Summary of Results
The graphs below showcase the training and test accuracies over epochs for each model:

<img src="https://github.com/user-attachments/assets/a3e054b5-3e7d-48fc-b5f2-998a36a02052" width="49%">
<img src="https://github.com/user-attachments/assets/6b2a4f7b-8642-4a43-a7a3-6f5c0be77f62" width="49%">
<img src="https://github.com/user-attachments/assets/05362145-ceb3-4a46-b16c-53d4071e437f" width="49%">
<img src="https://github.com/user-attachments/assets/973a6afe-3e1f-457a-912c-b032c6e6d4af" width="49%">
<img src="https://github.com/user-attachments/assets/c07ff62e-5ff9-489d-a5d3-c6de02a22b12" width="49%">

The table below summarizes the final training and validation accuracies for each model:

| Model                     | Train Accuracy | Validation Accuracy |
|---------------------------|----------------|---------------------|
| Baseline                  | 0.932076       | 0.892121            |
| Dropout                   | 0.889422       | 0.897108            |
| L2 Regularization         | 0.843269       | 0.850731            |
| Optimized L2 Regularization | 0.907638     | 0.887467            |
| Batch Normalization       | 0.944757       | 0.902427            |

## Analysis and Conclusions

* **Baseline Model:** Achieved a high training accuracy of 93.2%, but the validation accuracy was slightly lower at 89.2%, indicating some overfitting.
* **Dropout Model:** Reduced overfitting compared to the baseline, with a slightly higher validation accuracy (89.7%) than its training accuracy (88.9%), suggesting that dropout effectively improved generalization.
* **L2 Regularization:** Showed lower performance in both training (84.3%) and validation (85.1%) accuracies, indicating that the chosen regularization strength might have been too high, leading to underfitting.
* **Optimized L2 Regularization Model:** Improved over the initial L2 regularization, achieving 90.8% training accuracy and 88.7% validation accuracy. This suggests that fine-tuning the regularization parameters positively impacted model performance, though it still fell short of the baseline in validation.
* **Batch Normalization:** Achieved the highest performance among all models, with a training accuracy of 94.5% and validation accuracy of 90.2%. This indicates that batch normalization was effective in improving model stability and accelerating training.

Overall, our Batch Normalization model was the best-performing, giving us back the highest accuracy on both the training and validation sets. Our Optimized L2 Regularization Model showed improvements over the basic L2 approach, but still did not surpass the baseline model’s performance, which highlights the challenges in balancing regularization strength and training duration of these models.

## References

1. **Gradient-Based Learning Applied to Document Recognition**  
   Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.  
   *Proceedings of the IEEE*, 86(11):2278-2324, 1998.  
   [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  

2. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**  
   Sergey Ioffe and Christian Szegedy.  
   *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 2015.  
   [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)  

3. **Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms**  
   Han Xiao, Kashif Rasul, and Roland Vollgraf.  
   *arXiv preprint arXiv:1708.07747*, 2017.  
   [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)  

