# Scalable Machine Learning and Deep Learning
KTH course ID2223 repository featuring the two labs

## Lab 2
This lab features the whisper network trained to transcribe Dutch. The model is used to transcribe audio retrieved from user
microphone input, a file upload or a youtube link.

#### Model-centric improvements
In order to improve performance, one ought to simply train more, i.e. increasing the number of steps. Another tuning step could be 
changing the learning rate to a smaller initial value as it decays quickly in the original run.

#### Data-centric improvements
Using more data for training would also improve the performance generally speaking. The data source for common_voice in Dutch is large,
as for the initial implementation only the first 30% was used. Therefore, using more should result in better results.

#### Model and Huggingface application
The models can be found here:
Initial model: https://huggingface.co/AlexMo/FIFA_WC22_WINNER_LANGUAGE_MODEL
Improved model: https://huggingface.co/AlexMo/improved_whisper_model

The resulting application can be found here:
https://huggingface.co/spaces/AlexMo/audio_summarizer