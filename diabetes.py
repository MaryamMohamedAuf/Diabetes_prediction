#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
file_path = r"C:\Users\win10\Downloads\pima-indians-diabetes.data.csv"
data = np.loadtxt(file_path, delimiter=',')
input_data = data[:, 0:8]
output_data = data[:,8]
#build ML model
get_ipython().system('pip install tensorflow')


# In[22]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



# In[ ]:


set configuration


# In[23]:


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[ ]:


train the model


# In[24]:


model.fit(input_data, output_data, epochs = 150 , batch_size = 10)


# In[27]:


accuracy = model.evaluate (input_data,output_data, verbose = 0)
accuracy


# In[29]:


predictions = (model.predict(input_data) > 0.5) . astype(int)


# In[34]:


def display(num):
    if num == 0:
        return "no disease";
    else:
              return " disease";
for i in range(10):
	print(" %s => %s (expected %s) " % (input_data[i].tolist(),display(predictions[i]),display(output_data[i])))


