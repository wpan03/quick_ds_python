# Reusable Tabular Data Workflow :ocean:

## Vision :relaxed: 

Over time, I found that I need very similar codes across multiple data science projects. Thus, I decide to put codes that can solve those needs in one place to save myself time. It will be great if this can also make your work a little bit easier! 

My understanding for data science and coding keep changing and thus the code in this repo will be constantly updated. Any contributions, suggestions, and feedbacks will be really appreciated！

## Directory Structure :scroll:

```
├── README.md                         <- You are here
├── src/                              <- Source module for the project 
├── nice_things/                      <- Some codes that can be directly copied and used
├── *.ipynb                           <- notebooks that demonstrate the usage functions in src
├── requirement.txt                   <- describe the python package version when this code is developed
```

## Tour :steam_locomotive:

This section gives an overview of the purpose of each file.

### src

This folder contains modularized functions that can be easily reused. 

+ `eda.py`: contain codes that is helpful for exploring data analysis.
+ `model_supervised.py`: contain codes that develop a supervised learning model, including codes for hyper-parameter tuning.
+ `evaluate.py`: contain functions that evaluate the performance of a supervised learning model.
+ `explain.py`: contain codes that explain why a model makes certain prediction.
+ `model_cluster.py`: contain functions that simplify the process of developing and analyzing cluster models, especially for KMeans. 

### nice_things

This folder contain some codes that are not in a function, but can be handy for copying when developing. 

+ `tune_grid.py`: contain predefined parameter grids that can be a good start point for doing hyper-parameter tuning.

### Jupyter Notebooks

+ `supervised_clf_demo.ipynb`: demonstrates how the function in this repo can help various stages when developing a classification model. 
+ `supervised_reg_demo.ipynb`: demonstrates how the function in this repo can help various stages when developing a regression model. 
+ `cluster_demo.ipynb`: demonstrates how the function in this repo can help developing a clustering model. 

