from taipy.gui import Gui, notify
from datetime import date
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib as plt
import plotly.graph_objs as go
import json

# Parameters for retrieving the stock data
start_date = "2015-01-01"
end_date = date.today().strftime("%Y-%m-%d")
# selected_stock = 'AAPL'
selected_player = ''
n_years = 1


# def get_stock_data(ticker, start, end):
def get_basketball_data(selected_player, start, end):
    ticker_data = yf.download(selected_player, start, end)  # downloading the stock data from START to TODAY
    ticker_data.reset_index(inplace=True)  # put date in the first column
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date']).dt.tz_localize(None)
    return ticker_data

def get_data_from_range(state):
    print("GENERATING HIST DATA")
    start_date = state.start_date if type(state.start_date)==str else state.start_date.strftime("%Y-%m-%d")
    end_date = state.end_date if type(state.end_date)==str else state.end_date.strftime("%Y-%m-%d")

    state.data = get_basketball_data(state.selected_player, start_date, end_date)
    if len(state.data) == 0:
        notify(state, "error", f"Not able to download data {state.selected_stock} from {start_date} to {end_date}")
        return
    notify(state, 's', 'Historical data has been updated!')
    notify(state, 'w', 'Deleting previous predictions...')
    state.forecast = pd.DataFrame(columns=['Date', 'Lower', 'Upper'])



# def generate_forecast_data(data, n_years):
def generate_hotspot_data(player_name, n_years, data):
    """
    Generates hotspot data for a given player over a specified number of years.

    :param player_name: The name of the player.
    :param n_years: The number of recent years to include.
    :param data: A pandas DataFrame containing shot data.
    :return: JSON representation of Plotly graph data.
    """

    # Assume 'Season' is an integer or can be converted to one representing the year
    # And 'data' is sorted in ascending order by 'Season'
    recent_season = data['Season'].max()  # Get the most recent season
    start_season = recent_season - n_years + 1  # Calculate the start season
    
    # Filter the data for the specified player and the last n years
    filtered_data = data[(data['PlayerName'] == player_name) & (data['Season'] >= start_season)]
    
    # Generate the Plotly graph object
    plot_data = go.Scatter(
        x=filtered_data['X'], 
        y=filtered_data['Y'], 
        mode='markers',
        marker=dict(size=10, color='rgba(255, 0, 0, .8)'),  # Customize as needed
        name=player_name
    )

    # Convert the Plotly graph object to JSON
    graphJSON = json.dumps([plot_data], cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def forecast_display(state):
    notify(state, 'i', 'Predicting...')
    state.forecast = generate_hotspot_data(state.data, state.n_years)
    notify(state, 's', 'Prediction done! Forecast data has been updated!')



#### Getting the data, make initial forcast and build a front end web-app with Taipy GUI
data = get_basketball_data(selected_player, start_date, end_date)
forecast = generate_hotspot_data(data, n_years)

show_dialog = False

partial_md = "<|{forecast}|table|>"
dialog_md = "<|{show_dialog}|dialog|partial={partial}|title=Forecast Data|on_action={lambda state: state.assign('show_dialog', False)}|>"

page = dialog_md + """<|toggle|theme|>
<|container|
# Basketball HotShot KMean Cluster Analyzer Dashboard

<|layout|columns=1 2 1|gap=40px|class_name=card p2|

<dates|
#### Selected **Period**{: .color-primary}

From:
<|{start_date}|date|on_change=get_data_from_range|>  

To:
<|{end_date}|date|on_change=get_data_from_range|> 
|dates>

<ticker|
#### Selected **Player**{: .color-primary}

Please enter a valid Basketball Player: 
<|{selected_player}|input|label=Stock|on_action=get_data_from_range|> 


or choose a popular one

<|{selected_stock}|toggle|lov=Trae Young; Chris Paul; Seth Curry; Russell Westbrook|on_change=get_data_from_range|>
|ticker>

<years|
#### Prediction **years**{: .color-primary}
Select number of prediction years: <|{n_years}|>  
<|{n_years}|slider|min=1|max=5|>  

<|PREDICT|button|on_action=forecast_display|class_name={'plain' if len(forecast)==0 else ''}|>
|years>

|>


<|Historical Data|expandable|expanded=False|
<|layout|columns=1 1|
<|
### Historical **closing**{: .color-primary} price
<|{data}|chart|mode=line|x=Date|y[1]=Open|y[2]=Close|>
|>

<|
### Historical **daily**{: .color-primary} trading volume
<|{data}|chart|mode=line|x=Date|y=Volume|>
|>
|>

### **Whole**{: .color-primary} historical data: <|{selected_stock}|text|raw|>
<|{data}|table|>

<br/>
|>


### **Forecast**{: .color-primary} Data
<|{forecast}|chart|mode=line|x=Date|y[1]=Lower|y[2]=Upper|>

<br/>


<|More info|button|on_action={lambda s: s.assign("show_dialog", True)}|>
{: .text-center}
|>

<br/>
"""


# Run Taipy GUI
gui = Gui(page)
partial = gui.add_partial(partial_md)
gui.run(dark_mode=False, title="Basketball HotShot KMean Cluster Analyzer")
