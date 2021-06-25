# Emotion-Recognition-API

Converting a Jupyter notebook used to train an emotion recognition model into an API using Cuttle.

## 🚀 Installation

```pip install cuttle```

## Initialise Cuttle

Initialise cuttle in the same folder containing your Jupyter Notebook.

```cuttle init```

This step creates a cuttle.json file in the same directory.

## Create cuttle environment

In this step, specify the environment name, platform and the transformer to be used.

```cuttle create```

Notice the updated cuttle.json after this step.

## Adding config

Let's add the cell scoped config and line scoped config as seen in [Notebook](Emotion_recogniser.ipynb)

```#cuttle-environment-set-config <environment-name> method=POST route=<route> response=<variable>```

```#cuttle-environment-assign <environment-name> <dependancy>```

## Cuttle transform

Use the environment name specified in the previous step.

```cuttle transform <environment-name>```

TA-DA! You should now see an output folder created in the same repository containing a sub directory with the environment name. This folder contains the transformed file.

## ⚡️ Steps to test

```python output/<environment-name>/main.py```

Your code is now running on the flask server. By default this port is localhost:5000. You can now send a file to localhost:5000/<route> to test your model.
