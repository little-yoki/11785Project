
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


filename = 'nyu_data.npy'
print('Loading data...')
inputs = np.load(filename)
depths = inputs[()]['depths']
labels = inputs[()]['labels']
images = inputs[()]['images']
print('Data loaded succeeded!')


# In[3]:

labels = np.transpose(labels, (2, 0, 1))  # N x H x W

# In[4]:


label_unique = [np.unique(image) for image in labels]
class_unique, count = np.unique(np.concatenate(label_unique), return_counts = True)
class_sorted = [x for _,x in sorted(zip(count,class_unique))]
count_sorted = [y for y,_ in sorted(zip(count,class_unique))]
class_sorted.reverse()
count_sorted.reverse()


# In[5]:


class_small = class_sorted[:10] # >380
class_medium = class_sorted[:45] # >100
class_big = class_sorted[:178] # > 10
labels_small = np.copy(labels)
labels_medium = np.copy(labels)
labels_big = np.copy(labels)


# In[6]:


# small image label
for i in range(len(labels_small)):
    labels_small[i] = np.where(np.isin(labels_small[i], class_small), labels_small[i], 0)

# medium image label
for i in range(len(labels_medium)):
    labels_medium[i] = np.where(np.isin(labels_medium[i], class_medium), labels_medium[i], 0)
    
# big image label
for i in range(len(labels_big)):
    labels_big[i] = np.where(np.isin(labels_big[i], class_big), labels_big[i], 0)


# In[7]:


images_small = np.copy(images)
images_medium = np.copy(images)
images_big = np.copy(images)
depths_small = np.copy(depths)
depths_medium = np.copy(depths)
depths_big = np.copy(depths)


# In[9]:


dataset_small = {'images': images_small, 'depths': depths_small, 'labels': labels_small, 'classes': class_small}
dataset_medium = {'images': images_medium, 'depths': depths_medium, 'labels': labels_medium, 'classes': class_medium}
dataset_big = {'images': images_big, 'depths': depths_big, 'labels': labels_big, 'classes': class_big}
np.save('dataset_small.npy', dataset_small)
np.save('dataset_medium.npy', dataset_medium)
np.save('dataset_big.npy', dataset_big)

