import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np

fig = go.Figure()

import sympy as sp
X = sp.symbols('X')

def flip_fun(exp_fun, x_values):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    # convert x values to float type
    x_values = np.array(x_values, dtype=float)
    
    y_values = []
    for val in x_values:
          y_values.append(exp_fun.subs(X, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(exp_fun)) )

    fliped_fun = exp_fun.subs(X, -X)

    y_values = []
    for val in x_values:
          y_values.append(fliped_fun.subs(X, val))
    
    y_values = np.array(y_values, dtype=float)

    fliped_fun = exp_fun.subs(X, -X)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(fliped_fun)))

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Flip Function', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")

    # Update the figure:
    fig.show()

     
def shift_fun(exp_fun, x_values, change):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    x_values = np.array(x_values, dtype=float)
    y_values = []
    for val in x_values:
          y_values.append(exp_fun.subs(X, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(exp_fun)) )

    if isinstance(change, sp.Basic):
        modified_fun = exp_fun.subs(X, change)

        y_values = []
        for val in x_values:
            y_values.append(modified_fun.subs(X, val))
            
        y_values = np.array(y_values, dtype=float)
        fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(modified_fun) + str(change)) )
        
    else:
         y_values = [y + change for y in y_values]
         
         y_values = np.array(y_values, dtype=float)
         fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(exp_fun) + f"{change}" ) )

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Shift Function', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")

    # Update the figure:
    fig.show()

def compare_fun(exp_fun, modified_fun, x_values):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    x_values = np.array(x_values, dtype=float)
    y_values = []
    for val in x_values:
          y_values.append(exp_fun.subs(X, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(exp_fun)) )

    y_values = []
    for val in x_values:
        y_values.append(modified_fun.subs(X, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name= str(modified_fun)) )

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Compare Functions', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")

    # Update the figure:
    fig.show()


     
     
     
def plot_fun(exp_fun, x_values):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    y_values = []
    for val in x_values:
          y_values.append(exp_fun.subs(X, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers' ) )

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Function Plot', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")


    # Update the figure:
    fig.show()
