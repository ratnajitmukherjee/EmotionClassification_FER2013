import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('C:/Users/PC/Documents/FacialExpression/fer2013/fer2013.csv')
#df.head()
#df.groupby(['Usage', 'emotion'])['emotion'].count().iloc[14:21].plot(kind='bar')
#plt.xticks(np.arange(7), ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'))
#plt.ylabel('Number of samples')
#plt.title('Distribution of training data')
#plt.show()

##df.groupby(['emotion'])['emotion'].count().plot(kind='bar')
##plt.xticks(np.arange(7), ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'))
##plt.ylabel('Number of samples')
##plt.title('Distribution of total (train + val + test) data')
##plt.show()


ax = df['Usage'].value_counts().sort_index().plot(kind='barh', figsize=(10,7))
ax.set_alpha(0.8)
ax.set_title("Total data (train + val + test) distribution")
ax.set_xlabel("Number of samples");
#ax.set_yticks(('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'))

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=12,
color='blue')

# invert for largest on top 
ax.invert_yaxis()
#ax.set_yticklabels(['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
ax.set_yticklabels(['Validation', 'Test', 'Training'])
plt.show()
