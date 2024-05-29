import os
from dash import dash_table
from dash import Dash
from jupyter_dash import JupyterDash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

filtered_cardiac_interventions_ROC = pd.read_csv('filtered_cardiac_interventions_ROC.csv')
# Check for any None values and handle them
filtered_cardiac_interventions_ROC.dropna(subset=['Latitude Intervention', 'Longitude Intervention', 'deltaT', 'Postal Code'], inplace=True)

filtered_cardiac_interventions_td = pd.read_csv('filtered_cardiac_interventions_td.csv')

# Perform the merge operation
filtered_cardiac_interventions = pd.merge(
    filtered_cardiac_interventions_ROC,
    filtered_cardiac_interventions_td[['Mission ID', 'T_d']],
    on='Mission ID',
    how='left'  # Use 'left' to keep all rows of filtered_cardiac_interventions and only matching rows from filtered_cardiac_interventions_td
)

filtered_cardiac_interventions['T_d'] = filtered_cardiac_interventions['T_d']*2

# Drop the existing deltaT column if it exists
if 'deltaT' in filtered_cardiac_interventions.columns:
    filtered_cardiac_interventions = filtered_cardiac_interventions.drop(columns=['deltaT'])

# Define a function that implements the decision tree logic
def calculate_columns(row):
    T3_T0 = row['T3_T0_difference_minutes']
    T_max_V = row['T Max (V)']
    T_max_VD = row['T Max (V+D)']
    T_d = row['T_d']
    
    if T3_T0 < T_max_V:
        deltaT = 0
        dueto_AED = 0
    elif T3_T0 > T_max_V and T3_T0 < T_max_VD:
        if T_d < T_max_V:
            deltaT = 0
            dueto_AED = 0
        else:
            deltaT = min(T_d-T_max_V, T3_T0-T_max_V)
            dueto_AED = 1
    elif T3_T0 > T_max_VD:
        if T_d < T_max_V:
            deltaT = T3_T0 - T_max_VD
            dueto_AED = 0
        else:
            deltaT = T3_T0 - T_max_V
            dueto_AED = 0
    
    return pd.Series([deltaT, dueto_AED], index=['deltaT', 'dueto_AED'])

# Apply the function to each row
filtered_cardiac_interventions[['deltaT', 'dueto_AED']] = filtered_cardiac_interventions.apply(calculate_columns, axis=1)

filtered_cardiac_interventions.head(5)

# URL to the raw CSV file
url = "https://github.com/jief/zipcode-belgium/raw/master/zipcode-belgium.csv"

# Column names
column_names = ['Postal Code', 'Commune Name', 'Longitude Commune', 'Latitude Commune']

# Read the CSV file into a DataFrame without specifying a header
communes = pd.read_csv(url, header=None)

# Manually set the column names
communes.columns = column_names

# Ensure no empty spaces and reset index
communes['Commune Name'] = communes['Commune Name'].str.strip()
communes = communes.reset_index(drop=True)

# Define the function to determine the province
def determine_province(postal_code):
    if 2000 <= postal_code <= 2999:
        return "Antwerp"
    elif 1000 <= postal_code <= 1299:
        return "Brussels"
    elif 9000 <= postal_code <= 9999:
        return "East Flanders"
    elif (1500 <= postal_code <= 1999) or (3000 <= postal_code <= 3499):
        return "Flemish Brabant"
    elif (6000 <= postal_code <= 6599) or (7000 <= postal_code <= 7999):
        return "Hainaut"
    elif 4000 <= postal_code <= 4999:
        return "Liege"
    elif 3500 <= postal_code <= 3999:
        return "Limburg"
    elif 6600 <= postal_code <= 7000:
        return "Luxembourg"
    elif 5000 <= postal_code <= 5999:
        return "Namur"
    elif 1300 <= postal_code <= 1499:
        return "Walloon Brabant"
    elif 8000 <= postal_code <= 8999:
        return "West Flanders"
    else:
        return "Unknown"

# Define the function to determine the region
def determine_region(postal_code):
    if 1000 <= postal_code <= 1299:
        return "Brussels"
    elif (1500 <= postal_code <= 3999) or (8000 <= postal_code <= 9999):
        return "Flanders"
    else:
        return "Wallonia"

# Convert Postal Code to int for correct processing
communes['Postal Code'] = communes['Postal Code'].astype(int)

# Group by 'Postal Code' and aggregate
communes_aggregated = communes.groupby('Postal Code').agg({
    'Commune Name': lambda x: ' / '.join(x),
    'Longitude Commune': 'mean',
    'Latitude Commune': 'mean'
}).reset_index()

# Apply the functions to create the new columns on the aggregated DataFrame
communes_aggregated['Province'] = communes_aggregated['Postal Code'].apply(determine_province)
communes_aggregated['Region'] = communes_aggregated['Postal Code'].apply(determine_region)

# Save to CSV (optional)
communes_aggregated.to_csv('communes_list.csv', index=False)

# Add an intermediate column to filter rows where deltaT > 0
filtered_cardiac_interventions['is_deltaT_positive'] = filtered_cardiac_interventions['deltaT'] > 0

# Group by 'Postal Code' and compute necessary statistics
aed_potential = filtered_cardiac_interventions.loc[filtered_cardiac_interventions['is_deltaT_positive']].groupby('Postal Code').agg({
    'dueto_AED': 'mean',
    'is_deltaT_positive': 'sum'
}).rename(columns={
    'dueto_AED': 'AED_Potential_Ratio',
    'is_deltaT_positive': 'deltaT_positive_count'
}).reset_index()

# Calculate 'AED Potential' as the ratio
aed_potential['AED Potential'] = aed_potential['AED_Potential_Ratio']

# Merge this aed_potential with the averages DataFrame
averages = filtered_cardiac_interventions.groupby('Postal Code').agg({
    'deltaT': 'mean',
    'Postal Code': 'size'
}).rename(columns={'Postal Code': 'count'}).reset_index()

# Merge averages with aed_potential
averages = averages.merge(aed_potential[['Postal Code', 'AED Potential']], on='Postal Code', how='left')

# Fill NaNs in 'AED Potential' where deltaT is 0
averages['AED Potential'] = averages['AED Potential'].fillna(0)

# Round to the first 2 cyphers after comma
averages['AED Potential'] = averages['AED Potential'] * 100
averages['AED Potential'] = averages['AED Potential'].round(2)
averages.rename(columns={'AED Potential': 'AED Potential [%]'}, inplace=True)

# Merge with the communes DataFrame to add the additional columns
averages = averages.merge(communes_aggregated, on='Postal Code', how='left')

# Determine the max value of deltaT for color range
max_deltaT = averages['deltaT'].max()

app = Dash(__name__)
app.title = "Cardiac Interventions Rankings"

# Manually scale sizes between 50 and 200
size_min = 20
size_max = 300
count_min = averages['count'].min()
count_max = averages['count'].max()

# Scale the count values to the desired size range
if count_max != count_min:
    averages['scaled_size'] = ((averages['count'] - count_min) / (count_max - count_min)) * (size_max - size_min) + size_min
else:
    averages['scaled_size'] = size_min

# Custom color scale from green to red
color_scale = [
    [0, "green"],
    [1, "red"]
]

# Map creation function
def create_figure(df):
        
        hover_texts = df['Commune Name'].apply(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else str(x))
        # Format numbers to two decimal places
        df['deltaT'] = df['deltaT'].round(2)

        fig = px.scatter_mapbox(
            df,
            lat='Latitude Commune',
            lon='Longitude Commune',
            size='count',
            color='deltaT',
            color_continuous_scale=color_scale,
            range_color=[0, 20],
            hover_name=hover_texts,
            hover_data={'Latitude Commune': False, 'Longitude Commune': False, 'count': True, 'deltaT': True, 'AED Potential [%]': True},
            labels={'deltaT': 'Time Delay', 'count':'Number of Observations'}
        )

        # Update marker sizes
        fig.update_traces(marker=dict(size=df['scaled_size']))

        fig.update_layout(
            mapbox_style='open-street-map',
            mapbox_zoom=6.8,
            mapbox_center={'lat': 50.8503, 'lon': 4.3517},
            font=dict(family="Arial")
        )
        return fig

# Create initial map figure
fig = create_figure(averages)

n = 5
k = 10

# Filter out provinces with `None` value to fix the dropdown error
valid_provinces = averages['Province'].dropna().unique()

# Format numbers to two decimal places
averages['deltaT'] = averages['deltaT'].round(2)

# App layout
app.layout = html.Div([
    html.H1('Intervention time delay* in case of cardiac arrest, average time per Commune', style={'font-family': 'Arial', 'text-align': 'center'}),
    html.H2('Interactive Map and Table, National and by Province', style={'font-family': 'Arial', 'text-align': 'center'}),
    dcc.Dropdown(
        id='province-filter',
        options=[{'label': province, 'value': province} for province in valid_provinces],
        multi=True,
        placeholder="Select Province(s)",
        style={'font-family': 'Arial'}
    ),
    html.Div([
        dcc.RadioItems(
            id='ordering-parameter',
            options=[
                {'label': 'Time Delay', 'value': 'deltaT'},
                {'label': 'AED Potential [%]', 'value': 'AED Potential [%]'}
            ],
            value='deltaT',  # Setting default value
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            style={'font-family': 'Arial', 'margin-top': '10px'}
        ),
    ]),
    html.Div([
        dcc.Graph(id='map', figure={'data': []}, style={'width': '65%', 'display': 'inline-block', 'height': '600px'}),
        dash_table.DataTable(
            id='table',
            columns=[
                {'name': 'Postal Code', 'id': 'Postal Code'},
                {'name': 'Commune Name(s)', 'id': 'Commune Name'},
                {'name': 'Time Delay', 'id': 'deltaT'},
                {'name': 'Number of Observations', 'id': 'count'},
                {'name': 'AED Potential [%]', 'id': 'AED Potential [%]'},
                {'name': 'Province', 'id':'Province'}
            ],
            style_table={'margin-top': '50px', 'width': '100%', 'height': '600px', 'overflowY': 'auto'},
            style_cell={'whiteSpace': 'normal','height': 'auto', 'font-family': 'Arial', 'font-size': '14px'},
            page_size=k,  # Ensure that the table can hold at least k rows
        )
    ], style={'display': 'flex'}),
    html.P("* With 'Time Delay' it is indicated the difference between the average historical time of intervention in the commune and a time of intervention (statistically individuated) which ensures a good chance of survival. For further information, please consult the documentation.", style={'font-family': 'Arial', 'margin-top': '10px', 'text-align': 'center'}),
    html.P("With the switch, it is possible to order the Communes either by Delay Time, or by AED Potential [%]: this last KPI reflects the amount of fatalities which would be avoided statistically (in percentage) if increasing the amount of available AEDs.", style={'font-family': 'Arial', 'margin-top': '10px', 'text-align': 'center'})
])

# Callbacks for interactions
@app.callback(
    [Output('map', 'figure'), Output('table', 'data')],
    [Input('province-filter', 'value'), Input('ordering-parameter', 'value')]
)
def update_map_and_table(selected_provinces, ordering_parameter):
    if selected_provinces:
        filtered_df = averages[averages['Province'].isin(selected_provinces)]
    else:
        filtered_df = averages

    filtered_df_min_n = filtered_df[filtered_df['count'] >= n]
    top_k_communes = filtered_df_min_n.nlargest(k, ordering_parameter)

    # Truncate the Commune Name for display in the table
    top_k_communes['Commune Name'] = top_k_communes['Commune Name'].apply(lambda x: (x[:30] + '...') if len(x) > 30 else x)

    # Update map
    fig = create_figure(filtered_df)

    # Center map on selected provinces with a zoom (set to average center here for simplicity)
    if selected_provinces:
        center_lat = filtered_df['Latitude Commune'].mean()
        center_lon = filtered_df['Longitude Commune'].mean()
        fig.update_layout(mapbox_center={'lat': center_lat, 'lon': center_lon}, mapbox_zoom=7.5)
    else:
        fig.update_layout(mapbox_center={'lat': 50.8503, 'lon': 4.3517}, mapbox_zoom=6.8)

    return fig, top_k_communes.to_dict('records')

# Function to run the app
server = app.server

if __name__ == '__main__':
        port = int(os.environ.get('PORT', 8050))
        app.run_server(debug=True, port=port, host='0.0.0.0')