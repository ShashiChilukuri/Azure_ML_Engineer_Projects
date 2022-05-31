# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data about the direct marketing campaigns of a Portuguese banking institution and this dataset is from [UCL ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The goal of this project is to classify if the client will subscribe a term deposit (variable y).

The best performing model was VotingEnsemble model produced by AutoML with accuracy of 91.76. Where as Logistic Regression model with HyperDrive support predicted with accuracy of 90.8%. 



## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The core component of this architecture includes

1. Getting the tabular dataset, 
2. Cleaning and preparing the data 
3. Applying Logistic Regression model
4. Passing this model through Hyperdrive 
5. Picking the best model

Steps one through three are part of train.py file and steps four and five are part of udacity_project.ipynb file. From execution standpoint, jupyter notebook will consume the train.py file to generate the best model by passing the Azure ML Hyperdrive to fine tune the parameters.

Data is related to Portuguese banking institution's phone marketing campains. It conisits of 20 features and a classification label (y) to classify if a client subscribed to a product (term deposit) or not. 

```python
# Features
age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed
# Label
y
```

These features are primarly bank's client data elements, client's social and economic context attributes and other attributes. 10 of these features are categorical in nature while the rest of 10 are numeric. The goal of this classification project is to predict if the client will subscribe (yes/no) a term deposit (variable y).

As part of step2, a number of prerequisite data cleaing setps are performed which include:

* Removeing NA's
* Encoding the target variable (label)
* Encoding categorical variables 
* Spliting the data (test size = 30%) into train and test

All these steps are performed using Pandas and SKLean libraries. Once data is cleaned and split, Logistic Regression model is applied with parameters C (Regularization) and Max_iter (Maximum number of iterations). And to finetune these parameters, used Azure's HyperDrive functionality which needs following data parameters to be supplied:

* Parameter Sampler
* Early stopping policy

**What are the benefits of the parameter sampler you chose?**

As part of this project, used [Random Parameter Sampler](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) to sample over a hyperparameter search space. In this algorithm,  parameter values are chosen from a set of discrete values or a distribution over a continuous range. Examples of functions you can use include: [choice](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#azureml-train-hyperdrive-parameter-expressions-choice), [uniform](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#azureml-train-hyperdrive-parameter-expressions-uniform), [loguniform](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#azureml-train-hyperdrive-parameter-expressions-loguniform),[normal](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#azureml-train-hyperdrive-parameter-expressions-normal), and [lognormal](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#azureml-train-hyperdrive-parameter-expressions-lognormal). Since the selection of parameters are random in nature, this is quicker than Grid Search sampler. 

**What are the benefits of the early stopping policy you chose?**

For the project, used [Bandit early termination policy](https://docs.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.sweep.banditpolicy?view=azure-ml-py). This policy is based on slack factor/slack amount and evaluation interval. Bandit policy ends a job when the primary metric isn't within the specified slack factor/slack amount of the most successful job. Key benefit of termination policy is that model hyperparameter tunning/training will stop when reached the goal and we dont need to wait for the entier training to complete. 

Reson for selecting Bandit early termination policy is that it allows to select an interval and once it exceeds the specified interval, this policy will ends the job. It easy to use and provides more flexibility over other stopping policies such as median stopping.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML (Automated Machine Learning) essentially automates all accepts of machine learning process i.e, feature engineering, selection of hyperparameters, model training etc.



The best model selected by the AutoML is VotingEnsemble model, which is based on 7 different ensemble models each with specific weightage as shown below:

```
'run_algorithm': 'VotingEnsemble',
'ensembled_iterations': '[11, 18, 0, 3, 9, 14, 13]',
'ensembled_algorithms': "['XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'SGD']",
'ensemble_weights': '[0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.36363636363636365, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091]',
```



## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

When comparing best model from HyperDrive vs AutoML, Accuracy of the models are pretty close with slight edge for AutoML model. While the best model Logistic Regression with HyperDrive got an accuracy of 90.8%, AutoML generated VotingEnsemble model predicted with accuracy of 91.76. Although difference between two models accuracy is small, there are a lot different in terms of implemetation and architecture. In case of AutoML, very few steps of code is needed when compared to HyperDrive model.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

In case of HyperDrive:

* Need to try with different classification models with HyperDrive. 
* Need to try with more wider range of hyper parameters in the HyperDrive

In case of AutoML,

* Need to try different sampling methods such as Grid search etc. need to be tested (as it is more comprehensive compared to Random Search)
* Need to try different termination policies and compare how it performes.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

Deletion of cluster can be done using code (as below) or going to compute page and deleting it manually. For this project, I have done it manually by visiting the Compute cluster page 

```
cpu_cluster.delete()
```

Here is the screenshot of deleting compute instance:

![image-20220531172217490](/Users/shashi/Documents/Job/Azure/Azure_ML_Engineer_Projects/Assets/:Users:shashi:Library:Application Support:typora-user-images:image-20220531172217490.png)
