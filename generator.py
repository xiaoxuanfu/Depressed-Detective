from tabgan.sampler import GANGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

ori_data = pd.read_csv("train_data.csv")

# Prepare data for GAN function
ori_train, test = train_test_split(ori_data, test_size=0.2)
train = ori_train.loc[:,ori_train.columns != 'label']
test = test.loc[:,test.columns != 'label']
target = ori_train.loc[:,ori_train.columns == 'label']

# Run GAN function to generate new synthetic data
new_train, new_target = GANGenerator().generate_data_pipe(train, target, test, )
generated_data = new_train.assign(label=new_target)
cols = generated_data.columns.tolist()
cols = cols[-1:]+cols[:-1]
generated_data = generated_data[cols]
full_data = pd.concat([ori_data, generated_data])

# Append new data to csv
full_data.to_csv('train_data_exteneded.csv', index=False)