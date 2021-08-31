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
├── pyproject.toml                    <- describe the python package version when this code is developed
```

## Virtual Environment :clipboard:

This section is mainly for development purpose. You can skip this section if you want to directly copy and use functions inside this repo. 

We use [poetry](https://python-poetry.org/) to manage python dependencies in this project. You can install poetry with the following command. 

### Install Poetry

The following is copied the [official guide of poetry](https://python-poetry.org/docs/master/). 

```zsh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```

 The installer installs the `poetry` tool to Poetry’s `bin` directory. This location depends on your system:

- `$HOME/.local/bin` for Unix
- `%APPDATA%\Python\Scripts` on Windows

If this directory is not on your `PATH`, you will need to add it manually if you want to invoke Poetry with simply `poetry`. What I did is that I add the following command in my `~/.zshrc`. (Mac OS)

```zsh
export PATH="$HOME/.local/bin:$PATH"
```

### Activate Virtual Environment

You can install all dependencies with the following.

```zsh
poetry install 
```

Now you can activate virtual environment with the following.

```zsh
poetry shell
```

You can exit the poetry shell by typing `exit` in the command line. 

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

+ `supervised_clf_demo.ipynb`: demonstrates how the function in this repo can help various stages when developing a classification model. [In Colab](https://colab.research.google.com/github/wpan03/quick_ds_python/blob/master/supervised_clf_demo.ipynb). 
+ `supervised_reg_demo.ipynb`: demonstrates how the function in this repo can help various stages when developing a regression model. 
+ `cluster_demo.ipynb`: demonstrates how the function in this repo can help developing a clustering model. 

