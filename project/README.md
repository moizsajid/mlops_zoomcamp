# Project: Car Price Prediction Service

For the MLOps Zoomcamp Project, the problem of price prediction is a classicial regression problem with many uses in for example housing, auto resale industry etc. For this project, the car price prediction is explored. The dataset used for the car price prediction is available from [Kaggle](https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/data) [1]. The dataset contains 19,238 records with both numerical and categorical attributes. The dataset can be viewed in the `data` folder. During processing, the data is divided into train, val and test subsets. The dataset preprocessing code was also from [Kaggle](https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/notebook) [2] but was adjusted as per the needs of the project. The final prediction file is containerized and makes use of .

An AWS EC2 instance is used for running `t2.micro` is used for training and deploying the model. Initially, the `preproces.py` file is excuted. This splits up the the dataset into train, val and test splits, and the corresponding `train.pkl`, `val.pkl` and `test.pkl` files in the `output` folder. After this, the `hpo.py` file is excuted. This file runs versions of the model with different hyperparameters. All the different configurationa along with their metrics are saved in the MLFlow server which is also running in the AWS instance. For running the MLFlow server, the following command is used:
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:DB_PASSWORD@DB_ENDPOINT:5432/mlflow_db --default-artifact-root s3://mlflow-models-mlopszoomcamp
```
In the command, the `DB_PASSWORD` and `DB_ENDPOINT` have to be specified. For security reasons, these values have been removed. The `register_model.py` files select the best MLFlows run from the execution of the `hpo.py` file. The best performing model is registered in MLFlow

For building the Docker image, the command below is used. This builds the Docker image from the `Dockerfile`.
```
docker build -t car-price-prediction-service:v1 .
```

After building the image, the web service in runned on cloud using the Docker run command. The Docker run command makes use of the `predict.py` file for serving the model. The `predict.py` file also preprocesses the data that is provided to it as input. The following command to run, the env variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `RUN_ID` need to set up already.
```
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e RUN_ID -it --rm -p 9696:9696 car-price-prediction-service:v1
```

Once the above the web service is running, run for example the `test.py` file. This file processes a sample JSON car record so that it can be feed to the model. For further testing, tests are included in the `tests` folder. GitHub Action (CI pipeline) and pre commit hooks makes use of these tests. The Terraform files are included in the `infrastructure` folder.

[1] [https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/data](https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/data)

[2] [https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/notebook](https://www.kaggle.com/code/giridharanp/car-price-prediction-ml/notebook)
