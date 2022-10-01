from webbrowser import GenericBrowser
from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from csv import writer

ori_data = pd.read_csv("emotion_output.csv")

ori_train, test = train_test_split(ori_data, test_size=0.2)
train = ori_train.loc[:,ori_train.columns != 'label']
test = test.loc[:,test.columns != 'label']
target = ori_train.loc[:,ori_train.columns == 'label']

new_train, new_target = GANGenerator().generate_data_pipe(train,target,test)
generated_data = new_train.assign(label=new_target)
cols = generated_data.columns.tolist()
cols = cols[-1:]+cols[:-1]
generated_data = generated_data[cols]

generated_data.to_csv('emotion_output.csv', mode='a', header=False, index=False)