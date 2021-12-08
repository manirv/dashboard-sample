import pandas as pd
import plotly.graph_objs as go
from scipy import stats
import numpy as np

from textwrap import dedent as d
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score


df2 = pd.read_csv('https://raw.githubusercontent.com/colaberry/DSin100days/master/data/Advertising.csv')

regr = skl_lm.LinearRegression()

X = df2[['Radio', 'TV']].as_matrix()
y = df2.Sales

regr.fit(X,y)
coef = regr.coef_
intercept = regr.intercept_

regr_model = regr
y_hat = regr.predict(X)
mse_ = mean_squared_error(y_hat,y)
print('mse is:', mse_)


def predict(tv, radio):
    data = pd.DataFrame({'TV': [tv], 'Radio': [radio]})
    return regr.predict(data)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

def get_zvalue(radio_max,tv_max):
    
    #coef = np.array([ 0.18799423,0.04575482])
    #intercept = 2.92109991241
    # Create a coordinate grid
    radio_range = max(50,int(round(radio_max)))
    tv_range = max(300, int(round(tv_max)))
    
    Radio = np.arange(0,radio_range)
    TV = np.arange(0,tv_range)
    
    B1, B2 = np.meshgrid(Radio, TV, indexing='xy')
    Z = np.zeros((TV.size, Radio.size))

    # Here is the place where we tilt and elevate the hyperplane
    for (i,j),v in np.ndenumerate(Z):
            Z[i,j] =(intercept + B1[i,j]*coef[0] + B2[i,j]*coef[1])

    return Z



text_content = ['''  > **About the Model & Algorithm**

              Linear regression is a supervised learning algorithm. Given a single feature, a line is fit that
              best predicts the independent variable. When many features are involved, a hyperplane is fit that
              minimizes the error between predicted values and the ground truth. 
                
              Given an input vector Xn = (X1, X2, ..., Xn) that we want to use to predict the output y, the
              regression equation is given by:

                   y= m*X + c

              The line that minimizes the Mean Squared Distance (MSE) is the best fit. Linear Regression is a
              statistical technique to determine that line. In this case, the regression model is a hyperplane 
              in a 3-dimensional space with Radio,TV and the predicted sales being the axes.
              
    > **Introduction to the Dataset**


              In this datadoc, we take a dataset of Advertising spends of various companies on 3 different media,
              namely, TV, Radio and Newspaper to analyze how the advertising spends affect the sales. This dataset 
              is part of the book "An Introduction to Statistical Learning with R" by Gareth James, et al. 
              This dataset contains about 200 records of advertising spends data of various companies. Each row 
              contains 4 columns with the input columns being the amount spent by a given unnamed company on TV, 
              Radio and Newspaper respectively for a given product. There is a 4th column which corresponds to the 
              sales generated for the company.
              
              
              Assuming that it is not possible for a company to increase sales of the product without advertising, 
              our job is to find out how to increase sales based on adjustments on advertising budgets. For 
              example, if the total buget is USD 1 million, and assuming we determine that the newspaper has no 
              impact beyond spending a basic USD 10,000, then we can allocate the budget on the other 2 media 
              namely Radio and TV according to  their impact on sales respectively.  Hence it becomes important 
              that we determing the correleation between the media advertising budget for each of TV, Radio and 
              Newspaper to the corresponding product sales.  ''',

              
'''               
   > 	**Exploratory Data Analysis of Sales Data**
   
              In this section we perform some Exploratory Data Analysis (or EDA) on the dataset with the Sales 
              from TV, Newspaper and Radio Advertisement spends and their corresponding sales for each product.
              
              
              Let us begin by drawing a pair plot or a scatter matrix to see how each input feature correlates to  
              the other. For example in the below plot, let us take a look at the plot of TV Vs Sales. This seems 
              to have a fairly linear relationship. This means that we could bring TV spend in to the mix for   
              performing linear regression. 
              Looking further, the relationship between the variable Radio against Sales, although the distribution   
              is more scattered compared to TV (vs Sales), we could sense some linearity underneath it.  
              Hence, we could take TV and Radio in a two feature multiple linear regression model. ''',

              
'''               
   > 	**Visual Representation of the Linear Regression Model**
   
              We use an sklearn LinearRegression based model to predict the Sales from TV and Radio Ad spends.
              
              Given below is the visual representation of the Linear Regression algorithm applied on the
              advertising dataset drawn using plotly interactive diagram. In this, we have considered two input
              parameters, namely, TV budget and Radio budget. In this 3D scatterplot, the 3rd axis is the 
              corrsponding sales figure for each set of Radio and TV budget.
              
              The points are represented by blue dots (or balls) within the 3D space represented by Radio(x),
              TV (y) and Sales (z) axes. The hyperplane represents the Linear Regression model that fits the 
              input predictor vector most effectively. This is drawn using the intercept value and the
              coefficients for Radio and TV advertisong budget variables based on the best Linear fit.
              
              Feel free to move the hyperplane around to different angles, to better understand the 
              Liner Regression plane and how it best fits the input variables.''']

def get_data_frame(): 
    return df2 


def getData():
    trace0 = go.Scatter3d(
                    y=df2['TV'],
                    x=df2['Radio'],
                    z=df2['Sales'],
                    #text='' + df2['TV'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    }
    )
    trace1 = go.Surface(z=get_zvalue(0,0), showscale=False, opacity=0.7)
    data = [trace0,trace1]
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
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
    return layout0

def getLayout2():
    layout0 = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
    return layout0

def getData2():
    text=[df2.loc[ k, 'Sales'] for k in range(len(df2))]
    trace0 = go.Splom(dimensions=[dict(label='Sales',
                                 values=df2['Sales']),
                            dict(label='TV',
                                 values=df2['TV']),
                            dict(label='Radio',
                                 values=df2['Radio']),
                            dict(label='Newspaper',
                                 values=df2['Newspaper'])],
                text=text,
                #default axes name assignment :
                #xaxes= ['x1','x2',  'x3'],
                #yaxes=  ['y1', 'y2', 'y3'], 
                marker=dict(size=7,
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
        )
    data = [trace0]
    return data


import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import app

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

#Methods for displaying Dataframe
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


layout = html.Div([
            
        html.H1(children='Predicting Sales using Regression'),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown(d(text_content[0])),
            ], style={'width': '80%', 'fontSize': 15}),
            html.Div([
            generate_table(get_data_frame())
            ], style={'width': '50%', 'fontSize': 15, 'margin-left': '10%'}),
            html.Div([
                dcc.Markdown(d(text_content[1])),
            ], style={'width': '80%', 'fontSize': 15}),
            html.Div([
            dcc.Graph(
            id='eda-graph',
            figure={
                'data': getData2(),
                'layout': getLayout2()
            }    
        )], style={'width': '60%', 'padding': '0 20'}),
            html.Div([
                dcc.Markdown(d(text_content[2])),
            ], style={'width': '80%', 'fontSize': 15}),
            html.Div([html.Img(id = 'cur_plot', src = '')],
             id='plot_div'),
            html.Div([
            dcc.Graph(
            id='advert-graph',
            figure={
                'data': getData(),
                'layout': getLayout()
            }    
        )], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'})

        ])
        ])

