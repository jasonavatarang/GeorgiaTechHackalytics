# Import modules and packages
from flask import (
    Flask,
    request,
    render_template,
    url_for,
    flash,
    redirect
)
import pickle
import numpy as np
from scipy.spatial import distance
import plotly.graph_objs as go
from plotly.offline import plot
import pandas as pd
from flask_socketio import SocketIO

all_data = pd.read_csv('./dataset/final_dataset.csv')

def cluster_user_preferences(data, first_name, last_name, shot_filter_num, lwr_date, upr_date):
    final_data_frame = pd.DataFrame()

  #filter by name first
   # Capitalize the first letter of each word
    print(first_name)
    print(last_name)
    capitalized_first_name = first_name.title()
    capitalized_last_name = last_name.title()
    name = capitalized_first_name + " " + capitalized_last_name
    #   name =get_player_name()
    #check if name exist in data_base
    # print(data.info())
    print(name)
    print(data[data["player"]==name])
    if len(data[data["player"]==name]) > 0:
        print("Player in dataset...filtering player stats")
        #filter the dataframe for the player
        player_frame = data[data["player"]==name]
        print(player_frame.head())
        #filter for shot selection:
        if shot_filter_num == "Both":
            print("No need to filter by point-shot")
            result = 0
        elif shot_filter_num == '2-pt':
            print('heeeeeyyeyy')
            result = 2

        elif shot_filter_num == '3-pt':
            result = 3
        #if you don't want to filter for shot types continue
        if result == 0:
            print ("no need to filter for shot type")
            player_frame_shot_filter = player_frame
            print(player_frame_shot_filter.head())
            #filter for 2 point or 3 point shots
        else:
            print(f"filtering for shot_type {shot_filter_num}")
            player_frame_shot_filter = player_frame[player_frame["shot_type"]==result]
            print(player_frame_shot_filter.head())
        #filter for days, range, or all data
        # date = filter_by_dates()
        print(lwr_date,'fasdf')
        lower_bound = ''.join(str(lwr_date).split('-'))
        upper_bound = ''.join(str(upr_date).split('-'))
        print(lower_bound, 'fhduy')
        print(upper_bound, 'fhddfsdfsuy')
        bounded_dates = (player_frame_shot_filter['match_id'] >= lower_bound) & (player_frame_shot_filter['match_id'] <= upper_bound)
        final_data_frame = player_frame_shot_filter[bounded_dates]
        print(final_data_frame.head())
        if final_data_frame is None:
            print('invalide date range')
        return final_data_frame

    else:
        print("Player not in dataset so we can't perform analysis :(" )
        return final_data_frame


import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Assuming cartesian_coordinates is defined somewhere
# For example: cartesian_coordinates = np.random.rand(100, 2)

def update_clustering(cartesian_coordinates,epsilon = 0.5, min_samples=5):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    print(f'{cartesian_coordinates} bro','dandafd')
    y_pred = dbscan.fit_predict(cartesian_coordinates)
    
    # Create a scatter plot for each cluster
    traces = []
    unique_labels = np.unique(y_pred)
    for label in unique_labels:
        cluster_coords = cartesian_coordinates[y_pred == label]
        if label == -1:
            # Noise points
            trace = go.Scatter(x=cartesian_coordinates['shotX'], y=cartesian_coordinates['shotY'], mode='markers',
                               marker=dict(color='rgba(217, 217, 217, 0.14)'), name='Noise')
        else:
            # Clustered points
            trace = go.Scatter(x=cartesian_coordinates['shotX'], y=cartesian_coordinates['shotY'], mode='markers',
                               marker=dict(size=10, opacity=0.7), name=f'Cluster {label}')
        traces.append(trace)

    # Create a layout with sliders for epsilon and min_samples
    layout = go.Layout(
        title=f'DBSCAN Clustering (epsilon={epsilon}, min_samples={min_samples})',
        xaxis=dict(title='X Coordinate'),
        yaxis=dict(title='Y Coordinate'),
        sliders=[{
            'pad': {"t": 30},
            'len': 0.4,
            'x': 0.5,
            'y': 0,
            'currentvalue': {
                'visible': True,
                'prefix': 'Epsilon:',
                'xanchor': 'right'
            },
            'transition': {'duration': 300},
            'steps': [{'label': str(i), 'method': 'restyle', 'args': ['epsilon', i]} for i in np.arange(0.1, 5.0, 0.1)]
        },
        {
            'pad': {"t": 30},
            'len': 0.4,
            'x': 0.5,
            'y': -0.05,
            'currentvalue': {
                'visible': True,
                'prefix': 'Min Samples:',
                'xanchor': 'right'
            },
            'transition': {'duration': 300},
            'steps': [{'label': str(i), 'method': 'restyle', 'args': ['min_samples', i]} for i in range(1, 10)]
        }]
    )
    
    # Combine traces and layout into a figure
    fig = go.Figure(data=traces, layout=layout)
    return fig


app = Flask(__name__)
app.config['SECRET_KEY'] = 'yeat@beat'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'
    
    if request.method == 'POST':
        input_val = request.form.to_dict()
        print(input_val)
        player_name = input_val['player_name']
        start_date = input_val['start-date']
        end_date = input_val['end-date']
        last_name = input_val['last_name']
        point = input_val['point']

        data = cluster_user_preferences(data = all_data, first_name = player_name, last_name = last_name, shot_filter_num= point, lwr_date=start_date, upr_date= end_date)

        # if player_name =="" or start_date =="" or end_date =="":
        #     flash('Invalid input: Input cannot be empty.', 'warning')
        #     return redirect(url_for('index'))
        # Example: Load your DataFrame here
        # For demonstration, assuming a DataFrame `data` exists
        # data = pd.DataFrame({
        #     'coord_x': [1,2],
        #     'coord_y': [1,2],
        #     'player_name': ['adf','fd']
        # })
        print(data.head())
        cartesian_coordinates =data[["shotX","shotY"]]
        fig = update_clustering(cartesian_coordinates=cartesian_coordinates)
        # Generate a Plotly figure
        # fig = go.Figure(data=go.Scatter(x=data['coord_x'], y=data['coord_y'], mode='markers', marker_color='#ff0000', text=data['player_name']))

        # Convert the figure to HTML
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        
        

        title = f"{player_name} hotspots from {start_date} to {end_date}"
        
        return render_template('index.html', plot_div=plot_div, title =title)

@app.route('/update_plot', methods=['POST'])
def update_plot():
    data = request.get_json()
    epsilon = float(data['epsilon'])
    min_samples = int(data['min_samples'])
    
    # Generate a new figure using the updated parameters
    fig = update_clustering(epsilon=epsilon, min_samples=min_samples, cartesian_coordinates=pd.DataFrame(np.random.rand(100, 2), columns=['shotX', 'shotY']))
    
    # Convert the figure to HTML and return it
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

@app.route('/filter', methods=['POST', 'GET'])
def filter():
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'
    
    if request.method == 'POST':
        input_val = request.form.to_dict()
        print(input_val)
        player_name = input_val['player_name']
        last_name = input_val['last_name']
        start_date = input_val['start-date']
        end_date = input_val['end-date']
        point = input_val['point']
        # if player_name =="" or start_date =="" or end_date =="":
        #     flash('Invalid input: Input cannot be empty.', 'warning')
        #     return redirect(url_for('index'))
        # Example: Load your DataFrame here
        # For demonstration, assuming a DataFrame `data` exists
        data = pd.DataFrame({
            'coord_x': [1,2],
            'coord_y': [1,2],
            'player_name': ['adf','fd']
        })


        # Generate a Plotly figure
        fig = go.Figure(data=go.Scatter(x=data['coord_x'], y=data['coord_y'], mode='markers', marker_color='#ff0000', text=data['player_name']))

        # Convert the figure to HTML
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)
        
        

        title = f"{player_name} {last_name} hotspots from {start_date} to {end_date} for {point} shots"
        
        return render_template('index.html', plot_div=plot_div, title=title)

@socketio.on('request update')
def handle_update(data):
    # Assume you have a function to process data and update the plot
    updated_title, plot_data = update_plot_and_title(data)
    
    # Emit updated data back to the client
    socketio.emit('update content', {'title': updated_title, 'plotData': plot_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
