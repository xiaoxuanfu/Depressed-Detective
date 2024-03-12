# Depressed-Detective
Official Codebase for Depressed Detective: monitoring of personnel well-being with computer vision and deep learning.
Submitted as part of the MLDA-EEE Deep Learning Week 2022 Hackathon.

The objective of this project is to serve as a preventative measure to detect workers facing burnout, especially in the context of Singapore striving towards Smart Nation, by leveraging computer vision and deep learning methods.

### Code Explanation

This repository consists of the following main files and directories:
(1). main.py
- Loads video data and obtain facial and audio features depending on arguments passed.
- Data is saved in ```train_data.csv```, consisting of 1 label, 7 facial and 12 audio features.
- Run test video to get prediction.

(2). summary.ipynb
- Provides summary of the entire pipeline from data collection, to data generation and model training and prediction.
- Offers in-depth analysis of quality of generated data and model evaluation.
- Shows test on custom video and output prediction.

(3). audio_analysis.py
- Read in video data and obtain audio features (relating to 12 pitch classes).
- Called in main.py if user inputs ```audio``` argument as True.

(4). generator.py
- Generate synthetic data from ```train_data.csv``` to obtain ```train_data_extended.csv```, containing twice the size.
- The main function used is from ```tabgan```, making use of Generative Adversarial Networks (GAN) to generate data.

(5). telebot.gs
- Consists of code for a Telegram bot that tells the user the people who are deemed as at-risk of depression.

(5). working_files
- Directory consisting of audio and video files for training and testing.

(6). metrics
- Consists of evaulation metrics to test the quality of generated data by ```generator.py```. 
- Results shown in ```summary.ipynb```.

### Demo Video

[![Demo Video](https://img.youtube.com/vi/LHZeXcXwLTY/0.jpg)](https://youtu.be/LHZeXcXwLTY)

### Example command

First, create a virtual environment for DAGAN and install the dependencies from ```requirements.txt```:

```shell
conda create -n depr_detective python=3.10
conda activate depr_detective
pip install -r requirements.txt
``` 
Then, run the main.py file

```shell
$ python main.py --test_file depressed --seq_len 25 --audio False --test False
```
