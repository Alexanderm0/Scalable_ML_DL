# Scalable_ML_DL
Projects for the KTH course ID2223 Scalable Machine Learning &amp; Deep Learning.

In this project we created a complete machine learning pipeline in serverless mode. 
In this way the pipeline was divided in different steps: Feature generation, Training, Batch Inference. 

![image](https://user-images.githubusercontent.com/53121540/203356492-745747e2-527b-4d9e-a1f1-932dbd3ab184.png)

We used Hopswork to store the features, the models and the predictions, instead we used Modal to run the different scripts. Finally we used Huggingface and Gradio to create the UI. 

#Project Goal
The goal of the project was to build a model that was able to learn from different features(sex, age, ...) wheather a passenger from the Titanic will survive or not.

##Implementation
First of all with the script **titanic-feature-pipeline.py** we take the dataset, process it and upload it with the feature in hopsworks. After this step we can run **titanic-training-pipeline.py** and train with KNN our model to predict if a passenger will survive or not. 
In the end we created an interactive app with Gradio, where users can insert value for passengers features and predict if they will survive or not.

We also created a new version for features generation called **titanic-feature-pipeline-daily.py** that creates randomly new passengers with their outcome. We subsequently use the script **titanic-batch-inference-pipeline.py** to predict the outcome of one of the new generated passanger. 
At this point with gradio we created a dashboard interface to show the last prediction and compared it to the actual outcome. Furthermore there is history of the last 5 prediction and a confusion matrix to sum up the score of our predictor.

###Links App UI
[Link Interactive UI](https://huggingface.co/spaces/tommyL99/titanic) 
[Link Dashboard UI](https://huggingface.co/spaces/tommyL99/titanic-monitoring) 

#Other Links
[Hopsworks](https://app.hopsworks.ai/app)
[Modal](https://modal.com/)
[Hugguing Face](https://huggingface.co/)




