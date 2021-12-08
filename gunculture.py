import numpy as np 
import pandas as pd 
from io import BytesIO
import six
from textwrap import dedent as d
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing 
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import seaborn as sns 
import pydotplus
location ='https://raw.githubusercontent.com/colaberry/datadocgen/master/data/fivethirtyEight-gun-violence-data.csv'
#load dataset
gun_violence_dataset_original= pd.read_csv(location)

# remove rows with NA 
gun_violence_dataset_original = gun_violence_dataset_original.dropna()
gun_violence_dataset_original = gun_violence_dataset_original.drop(['Unnamed: 0'], axis=1)

def val_count_to_percent(column): 
    """ Convert value counts of a dataframe column to percentages
    Input:
        column  - Nx1 array, Dataframe column
    Output: 
        percentages- Mx1 array, percentage of value counts for each variable  
    
    """
    return pd.value_counts(column)/(pd.value_counts(column).sum())*100 


val_count_to_percent(gun_violence_dataset_original['race'])

column  = gun_violence_dataset_original['race']
fig_width = 18 
fig_height = 15
save_fig = True
figname = 'race'

height = np.array(val_count_to_percent(column).values)

# sns.plt.figure(figsize=(fig_width,fig_height))
hue = list(val_count_to_percent(column).index)
mod_hue =[hue[0],hue[1],hue[2],'A/PI', 'NA/NAL'  ]
all_data = {'Feature labels':mod_hue , 'Percent of data': height, 'Race': hue}
df = pd.DataFrame(data = all_data)

text_content = '''## About the Algorithm

Decision trees are amazing tools to easily interpret a model since they split the data based on certain thresholds for each column of the data matrix. They are easy to train with few hyper-parameter and easy to interpret. They do have a major drawback, in that they tend to over fit the data. Despite this, as a first example of a machine learning algorithm, decision trees are intuitive and can give a valuable insight about the data which can be used to build better models. The goal of this analysis is to create a model using a decision tree on a dataset. 

Introduction to the dataset <br>

I will use the sklearn algorithm to train a decision tree. As you will see, many more lines are spent preparing the data for training rather than the actual training process (which really is just one line). The data set we are going to use can be found here -https://data.world/azel/gun-deaths-in-america. This data set is part of five thirty eight's gun deaths in America project. It contains a bunch of information about victims of gun violence. Each row of the dataset contains  the year and month of the shooting, the intent of the shooter, if cops were at the scene or not, the gender, age race and education level of the victim and finally the place where the shooting happened. There is specific information about whether the victim was Hispanic or not. We take this dataset and boil it down to predicting just one of two classes- were the victims of the shooting white or black? Why ignore the other victim classes? (There are 5 in total), firstly, the rest of the classes, as you will see, make up less than 11% of the dataset, secondly the goal is to build a simple binary classification model, for those who are interested, I would love to work with people build a more multi classification model for the whole CDC multiple causes of death dataset <br>


The plan for the analysis is as follows:
- Read and display the dataset to see what the relevant columns are.
- Encode certain categorical variables so the decision tree can be run on them.
- Plot some of the categorical variables to see how skewed they are.
- Drop rows containing non-white and non-black victims. 
- Create test and train sets.
- Train the decision tree. 
- Interpret the results of the tree- I will leave that as set of the question so that the interested reader can further get involved with understanding what the model represents

### Show class values for race

In order to assess the values in the race class, we need to retrieve  the value counts for each of the class variables. This number represents the number of victims of each race, and after retrieval, It is observed that a majority of the victims were either black or white. We can convert  these numbers into percentages and plot them for visualization as well.

From the bar plot below, it is obvious that majority of the data is for black or white victims. Here, I would like to point out that we can also make a multiclass problem by sampling the same number of rows as that of the Hispanic victims to create a 3 class classification problem. I chose to just stick with binary classification problem since this is more for demonstration purposes than hardcore analysis of the dataset. 

Personally, I believe that the complete version of this dataset requires proper analysis so that we can spot some trends on what kind of gun deaths or crimes are prevalent. Based on this if we can build a good prediction model,it can hopefully help law enforcement understand the nature of the problem better. This really is the whole point of the data driven approach. But I digress.
'''
import plotly.graph_objs as go

x = df['Race']
y = df['Percent of data']

def get_data_frame(): 
    return df 


def getData():
    trace0 = go.Bar(
            x=x,
            y=y,
            text=y,
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )
    data = [trace0]
    return data

def getLayout():
    layout0 = go.Layout(
                    scene = dict(
                        xaxis = dict(
                            title='Radio Spend(x)'),
                        yaxis = dict(
                            title='TV Spend(y)'),
                        zaxis = dict(
                            title='Sales(z)')),
                margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
                #legend={'x': 0, 'y': 1},
                hovermode='closest'
        )
    return layout0


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import app


layout = html.Div([

           html.H1(children='Gun Culture Analysis'),
           html.Div(className='row', children=[
        html.Div([
                dcc.Markdown(d(text_content)),
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
            dcc.Graph(
            id='advert-graph',
            figure={
                'data': getData()            }    
            )], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'})]),



            ])
