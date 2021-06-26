# Emotion-Recognition-API

Converting a Jupyter notebook used to train an emotion recognition model into an API using Cuttle.

## 🚀 Installation

Installing cuttle: ```pip install cuttle```

To install other dependencies: ```pip install -r requirements.txt```

## Initialise Cuttle

Initialise cuttle in the same folder containing your Jupyter Notebook. 
This step creates a cuttle.json file in the same directory.

```cuttle init```


## Create cuttle environment

In this step, specify the environment name, platform and the transformer to be used.

```cuttle create```

Notice the updated cuttle.json after this step.

## Adding config

Let's add the cell scoped config and line scoped config as seen in [Notebook](Emotion_recogniser.ipynb)

```#cuttle-environment-set-config emotion-rec method=POST route=/api/emotion response=output```

```#cuttle-environment-assign emotion-rec request.files['file']```

## Cuttle transform

Use the environment name specified in the previous step.

```cuttle transform emotion-rec```

TA-DA! You should now see an output folder created in the same repository containing a sub directory with the environment name. This folder contains the transformed file.

## ⚡️ Steps to test

```python output/emotion-rec/main.py```

Your code is now running on the flask server. By default this port is localhost:5000. You can now send a file to localhost:5000/api/emotion to test your model.
