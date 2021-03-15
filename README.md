# Credit Risk Modeling

This project intends to model credit risk by using machine learn algorithms. The dataset used was the [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). The solution is built with Python, Streamlit and Botorch, mainly. 


## Requirements 
* Docker
* Python
* Streamlit
* Ax Platform

## How to run 
**IMPORTANT**: Before running the project, be sure that RAM and Swap dedicated space for Docker is large enough. Suggested: Swap: 2GB, RAM: 2-4GB

In order to run the app, type the following command in your terminal to build and run the docker containers:
```
$ docker-compose up -d --build
```
And then you access the app on `http://localhost:8501/` 
