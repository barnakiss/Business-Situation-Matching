# Data analysis and wrangling
import pandas as pd
import numpy as np
from numpy import set_printoptions
import random as rnd
import math

from sklearn.linear_model import LinearRegression

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import CustomJS, ColumnDataSource, Slider, Select, GeoJSONDataSource
from bokeh.models import HoverTool, DatetimeTickFormatter, Legend, TapTool, BoxZoomTool, ResetTool
from bokeh.models import LinearAxis, DataRange1d, Plot
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.glyphs import Text
from bokeh.io import curdoc, show
from bokeh.models.widgets import Div
import bokeh
from bokeh.palettes import Category20_20, Plasma4, Plasma256

# Miscellaneous
import datetime
import sys
import os
import glob

# -----------------------------------------------------------------------

# To see all columns in a dataframe
pd.set_option('display.expand_frame_repr', False)

# =======================================================================

# Dates
START_DATE = '2016-06-30'
END_DATE = '2018-03-31'

PERIOD_START = '2017-03-31'
PERIOD_END = END_DATE

# b) Load dataset

# Acquire data
df = pd.read_excel('KPI_Export.xlsx')

# =======================================================================

# Compile country view of sunscription, churn & market share
# Limit us to metrics we use
df = df[(df['Metric'] == 'Avg Subscriptions in Q') | (df['Metric'] == 'Churn') | (
    df['Metric'] == 'Total Revenue') | (df['Metric'] == 'ARPU')]

# Get rid of unnecesarry data
# 'Currency' - everything is in USD
df = df.drop(['Region', 'Company ID', 'PMN Code', 'Currency'], axis=1)

# Melt df
mdf = df.melt(id_vars=['Country', 'Operator', 'Metric', 'Subscription Type'],
              var_name='Date',
              value_name='Value')

# -----------------------------------------------------------------------

# Convert Date into datetime
# Format: 'Q1 2002' --> '2002Q1'
mdf['Date'] = mdf['Date'].str[3:]+mdf['Date'].str[:2]
mdf['Date'] = pd.to_datetime(mdf['Date'])+pd.offsets.QuarterEnd(0)

# -----------------------------------------------------------------------

# Limit to last 2 years only
mdf = mdf[(mdf['Date'] >= START_DATE) & (mdf['Date'] <= END_DATE)]

# Take out rows with all NaNs
mdf.dropna(how='all')

# -----------------------------------------------------------------------

# Create list of all countries in mdf
all_countries = list(mdf['Country'].drop_duplicates().sort_values())

# -----------------------------------------------------------------------

# Create table with all operators to define colors for them
operators = mdf[['Country', 'Operator']
                ].drop_duplicates('Operator').reset_index()

# Calculate how many operators are in each country
num_ops = operators.groupby(['Country']).count()
operators = pd.merge(operators, num_ops, on=['Country'])
operators = operators.drop(['index_x', 'index_y'], axis=1)
# 'Number' = number of operator in the country
operators.columns = ['Country', 'Operator', 'Number']

# Prepare new column for the color
operators['Color'] = 0

max_j = 0
max_j_country = ''
previous_country = ''
for x, row in operators.iterrows():
    if row['Country'] != previous_country:
        previous_country = row['Country']
        j = 0
    if row['Operator'].find('T-Mobile') >= 0:
        operators.loc[x, 'Color'] = 'magenta'
    else:
        operators.loc[x, 'Color'] = Plasma256[j]
    if j > max_j:
        max_j = j
        max_j_country = row['Country']
    # To use full Plasma range of 256 colors
    j += math.floor(256/row['Number'])

# =======================================================================

# Calculate subscription
msu = mdf[mdf['Metric'] == 'Avg Subscriptions in Q']

# Combine all types of subscription
msu = pd.DataFrame(msu.groupby(['Country', 'Operator', 'Date'])['Value'].sum())

# -----------------------------------------------------------------------

# Pivot stacked data - country subscription pivot
su_pivot = msu.pivot_table(index=['Country', 'Operator'],
                           columns='Date',
                           values='Value',
                           aggfunc='sum')

# Replace zeros w/NaNs

su_pivot = su_pivot.replace({0: np.nan})

# Take out rows with all NaNs
su_pivot = su_pivot.dropna(how='all')

# =======================================================================

# Calculate market churn
mch = mdf[mdf['Metric'] == 'Churn']

# We don't need 'Metric' anymore as all values are Churn
mch = mch.drop(['Metric'], axis=1)

# -----------------------------------------------------------------------

# Pivot stacked data - country churn pivot
ch_pivot = mch.pivot_table(index=['Country', 'Operator'],
                           columns='Date',
                           values='Value',
                           aggfunc='sum')

# Replace zeros w/NaNs

ch_pivot = ch_pivot.replace({0: np.nan})

# Take out rows with all NaNs
ch_pivot = ch_pivot.dropna(how='all')

# =======================================================================

# Calculate market share - we're interested in Total Revenue only = Market Revenue
mr = mdf[mdf['Metric'] == 'Total Revenue']

# We don't need 'Metric' anymore as all values are Total Revenue
mr = mr.drop(['Metric'], axis=1)

# Country Total Revenues by Date
mrs = pd.DataFrame(mr.groupby(['Country', 'Date'])['Value'].sum())

# Merge on 'Country' & 'Date' --> melted Total Revenue (stacked format)
mtr = pd.merge(mr, mrs, on=['Country', 'Date'])
mtr.columns = ['Country', 'Operator', 'Subscription Type',
               'Date', 'Total Revenue', 'Country Total Revenue']
mtr['Market Share'] = mtr['Total Revenue']/mtr['Country Total Revenue']*100

# -----------------------------------------------------------------------

# Pivot stacked data - market share country pivot
ms_pivot = mtr.pivot_table(index=['Country', 'Operator'],
                           columns='Date',
                           values='Market Share',
                           aggfunc='sum')

# Replace zeros w/NaNs
ms_pivot = ms_pivot.replace({0: np.nan})

# Take out rows with all NaNs
ms_pivot = ms_pivot.dropna(how='all')

# =======================================================================

# Calculate ARPU
mch = mdf[mdf['Metric'] == 'ARPU']

# We don't need 'Metric' anymore as all values are Churn
mch = mch.drop(['Metric'], axis=1)

# -----------------------------------------------------------------------

# Pivot stacked data - country churn pivot
ar_pivot = mch.pivot_table(index=['Country', 'Operator'],
                           columns='Date',
                           values='Value',
                           aggfunc='sum')

# Replace zeros w/NaNs

ar_pivot = ar_pivot.replace({0: np.nan})

# Take out rows with all NaNs
ar_pivot = ar_pivot.dropna(how='all')

# =======================================================================

"""

- We evaluate country markets
- We are after situations which are similar to T-Mobile US in '13 Q1 - since we know it worked!
- We are in search for roles:
    - The 'Incumbent': the operator who may lose most from the Challenger's disruption
    - The 'Dropped Follower(s)': follower operator(s) with too small market share (marginal players - dropped from Benchmark charts)
    - The 'Last Follower': a follower operator on the market who's not too small, i.e. she is basically known by the customers
    - The 'Challenger': the operator who has good chance to disrupt the market


Categorisation:

- The 'Incumbent': the operator with the highest market share
- The 'Last Follower': the operator with enough market share (market share is the lowest compared to the period average but >=5%)
- The 'Dropped Follower(s)': disregarded operator(s) (market share is <15% @ the end of the period)


Business logic:
- Revenue stream:
    - Connectedness towards customers: Subscriptions
    - Customer relationship stability (noise, loss, wastes): Churn
    - Customer monetisation: Market Share


Criteria:

- The 'Last Follower' has lower subscription growth rate than the market average (in the last 4 quarters)
    --> She can capitalise on churn if she solves problem(s)
- The 'Last Follower' has higher churn ratio than the period average
    --> She can capitalise on churn if she solves problem(s)
- The 'Last Follower' is losing market share during the period (negative slope in the last 4 quarters)
    --> She has problem(s) why customers turn away from her. If she solves this problem she can keep & attract more customers
- The 'Last Follower' has lower ARPU (average revenue per user) rate than the market average (in the last 4 quarters)
    --> She has more cost sensitive customers
- The 'Incumbent' high churn ratio (higher than the period average)
    --> The 'Last Follower' can take more easily from the 'Incumbent'

If at least two criteria is met: The 'Last Follower' --> The 'Challenger'

"""

MARKET_SHARE_MINIMUM = 5  # 5%

bokeh_ref = '<link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.13.0.min.css" type="text/css" />'
bokeh_widget = [35, 0, 0, 18, 68, 92, 75, 85, 19, 71, 94, 72, 2, 83, 78, 76, 2, 1,
                29, 67, 14, 31, 16, 3, 22, 125, 86, 89, 22, 1, 7, 91, 6, 30, 0, 12, 67, 64, 83, 39]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

MAX_WIDTH = 1400
DROPDOWN_WIDTH = 300
HEADING_WIDTH = 1050
HEADING_HEIGHT = 100
DIV_WIDTH = 350
DIV_HEIGHT = 10
SCM_CHART_WIDTH = 350
SCM_CHART_HEIGHT = 250
TEXT1_WIDTH = 270
TEXT1_HEIGHT = 250

# -----------------------------------------------------------------------

chart_heading_plot = Plot(title=None, x_range=DataRange1d(), y_range=DataRange1d(),
                          plot_width=HEADING_WIDTH,
                          plot_height=HEADING_HEIGHT,
                          outline_line_color=None,  # Let chart grey frame disappear
                          toolbar_location=None)

# Text CHART_HEADING
CHART_HEADING = ['\nBusiness Situation Matching', '']
chart_heading_x = [0, 0]
chart_heading_y = [1, 0]  # Text goes vertical
chart_heading_source = ColumnDataSource(dict(x=chart_heading_x,
                                             y=chart_heading_y,
                                             text=CHART_HEADING))
chart_heading_glyph = Text(x="x", y="y", text='text',
                           text_align='center',
                           text_font_size="15pt",
                           text_baseline='top',
                           text_color='black')
chart_heading_plot.add_glyph(chart_heading_source, chart_heading_glyph)

# Text CHART_SUBHEADING
CHART_SUBHEADING = 'Pattern matching for disruptive business situations fitted to T-Mobile US Uncarrier disruption in 2013'
chart_subheading_x = [0]
chart_subheading_y = [0]
chart_heading_source = ColumnDataSource(dict(x=chart_subheading_x,
                                             y=chart_subheading_y,
                                             text=[CHART_SUBHEADING]))
chart_subheading_glyph = Text(x="x", y="y", text='text',
                              text_align='center',
                              text_font_size="12pt",
                              text_baseline='bottom',
                              text_color='black')
chart_heading_plot.add_glyph(chart_heading_source, chart_subheading_glyph)

# -----------------------------------------------------------------------

# Text explanation, below country selectionn

text1_chart = Div(text="""
<font face="Arial" size="1">
Definitions:<br>
- Incumbent: the market leader operator<br>
- (Potential) Challenger: a follower operator that has:<br>
&ensp;- lower subscription growth rate than market average (S)<br>
&ensp;- higher churn ratio than period average (C)<br>
&ensp;- decreasing market share during the period (M)<br><br>
Period: 2017 Q2-2018 Q1<br><br>
""", width=TEXT1_WIDTH, height=TEXT1_HEIGHT)

# -----------------------------------------------------------------------

# Apply model for each country
# DataFrame for the results
cv = pd.DataFrame()
for country in all_countries:
    # Initialisation of multiple variable
    outcome = last_follower = incumbent = ''
    lf_su_ratio = lf_churn_ratio = lf_market_share = lf_ms_slope = i_market_share = i_churn_ratio = lf_ar_ratio = np.nan

    # Find 'Last Follower' = lowest with Market Share > MARKET_SHARE_MINIMUM
    # Operators in the country in the Market Share dataset
    try:
        c_ms_ops = ms_pivot.loc[country, PERIOD_START:PERIOD_END].T.mean()

    except:
        outcome = 'Country not found'

    if outcome == '':
        # Get only rows where Market Share is not NaN
        c_ms_ops = c_ms_ops[~np.isnan(c_ms_ops)]

        if len(c_ms_ops) < 2:
            outcome = 'Not enough operators'
            # _=input('?')
        else:
            # Sort ascending
            c_ms_ops = c_ms_ops.sort_values()
            # Pick lowest
            for operator, market_share in c_ms_ops.iteritems():
                if market_share >= MARKET_SHARE_MINIMUM:
                    break
            if market_share < MARKET_SHARE_MINIMUM:
                outcome = f'Market Share minimum ({MARKET_SHARE_MINIMUM}%) not met'
            else:
                # 'Last Follower' identified
                last_follower = operator
                lf_market_share = market_share

                # -----------------------------------------------------------------------

                # Calculate Subscription slope of the country for the last 4 quarters
                su_model = LinearRegression(fit_intercept=True)
                try:
                    quarters = list(su_pivot.columns.astype(str))
                    period_quarters = quarters[quarters.index(
                        PERIOD_START):quarters.index(PERIOD_END)+1]
                    x = np.array([datetime.datetime.strptime(period_quarters[i], '%Y-%m-%d').toordinal()
                                  for i in range(len(period_quarters))]).reshape(-1, 1)
                    #x=np.array([datetime.datetime.strptime(str(su_pivot.columns.values[i])[:10],'%Y-%m-%d').toordinal() for i in range(4,8)]).reshape(-1, 1)
                    su_y = np.array(
                        list(su_pivot.loc[country, PERIOD_START:PERIOD_END].sum())).reshape(-1, 1)

                    su_model.fit(x, su_y)
                    ct_su_slope = su_model.coef_[0][0]
                except:
                    ct_su_slope = np.nan
                    lf_su_ratio = np.nan
                    outcome = 'Subscription slope cannot be calculated'

                # Calculate subscription slope of the 'Last Follower' for the last 4 quarters
                su_model_lf = LinearRegression(fit_intercept=True)
                try:
                    sulf_y = np.array(list(
                        su_pivot.loc[(country, last_follower), PERIOD_START:PERIOD_END])).reshape(-1, 1)
                    su_model_lf.fit(x, sulf_y)
                    lf_su_ratio = su_model_lf.coef_[0][0]/ct_su_slope
                except:
                    lf_su_ratio = np.nan
                    outcome = 'Last Follower Subscription slope cannot be calculated'

                # -----------------------------------------------------------------------

                # Calculate ARPU slope of the country for the last 4 quarters
                ar_model = LinearRegression(fit_intercept=True)
                try:
                    ar_y = np.array(
                        list(ar_pivot.loc[country, PERIOD_START:PERIOD_END].sum())).reshape(-1, 1)

                    ar_model.fit(x, ar_y)
                    ct_ar_slope = ar_model.coef_[0][0]
                except:
                    ct_ar_slope = np.nan
                    lf_ar_ratio = np.nan
                    outcome = 'ARPU slope cannot be calculated'

                # Calculate ARPU slope of the 'Last Follower' for the last 4 quarters
                ar_model_lf = LinearRegression(fit_intercept=True)
                try:
                    ysalf = np.array(list(
                        ar_pivot.loc[(country, last_follower), PERIOD_START:PERIOD_END])).reshape(-1, 1)
                    ar_model_lf.fit(x, yarlf)
                    lf_su_ratio = ar_model_lf.coef_[0][0]/ct_ar_slope
                except:
                    lf_ar_ratio = np.nan
                    outcome = 'Last Follower ARPU slope cannot be calculated'

                # -----------------------------------------------------------------------

                # Calculate Market Share slope of the 'Last Follower' for the last 4 quarters
                mslf_model = LinearRegression(fit_intercept=True)
                try:
                    mslf_y = np.array(list(
                        ms_pivot.loc[(country, last_follower), PERIOD_START:PERIOD_END])).reshape(-1, 1)
                    mslf_model.fit(x, mslf_y)
                    lf_ms_slope = mslf_model.coef_[0][0]
                except:
                    lf_ms_slope = np.nan
                    outcome = 'Last Follower Market Share slope cannot be calculated'

                # -----------------------------------------------------------------------

                # Calculate Churn ratio for the 'Last Follower'
                try:
                    lf_churn_ratio = ch_pivot.loc[(country, last_follower), PERIOD_START:PERIOD_END].mean() / \
                        ch_pivot.loc[country,
                                     PERIOD_START:PERIOD_END].mean().mean()
                except:
                    outcome = 'Last Follower Churn cannot be calculated'

                try:
                    # Find market leader --> incumbent
                    incumbent = list(c_ms_ops.keys())[-1]

                    # Incumbent Market Share
                    i_market_share = ms_pivot.loc[(
                        country, incumbent), PERIOD_START:PERIOD_END].mean()

                    # Calculate churn ratio for the 'Incumbent'
                    in_churn_ratio = ch_pivot.loc[(country, incumbent), PERIOD_START:PERIOD_END].mean(
                    )/ch_pivot.loc[country, PERIOD_START:PERIOD_END].mean().mean()
                    outcome = 'All parameters are calculated'

                except:
                    outcome = 'Market Leader Churn cannot be calculated'

    # Give points
    point1 = point2 = point3 = point4 = point5 = 0
    point_legend = ''
    if last_follower != '':
        if lf_su_ratio < 1:
            point1 = 1
            point_legend += 'S'
        if lf_churn_ratio > 1:
            point2 = 1
            point_legend += 'C'
        if lf_ms_slope < 0:
            point3 = 1
            point_legend += 'M'
        if lf_ar_ratio < 1:
            point4 = 1
            point_legend += 'A'
    if incumbent != '':
        if i_churn_ratio > 1:
            point5 = 1
    if len(point_legend) > 0:
        point_legend = ' - '+point_legend

    # Country view
    cv_data = {'Country': country,
               'Outcome': outcome,
               'Last Follower': last_follower,
               'Last Follower Subscription Ratio': lf_su_ratio,
               'Last Follower Churn Ratio': lf_churn_ratio,
               'Last Follower Market Share': lf_market_share,
               'Last Follower ARPU Ratio': lf_ar_ratio,
               'Last Follower Slope': lf_ms_slope,
               'Incumbent': incumbent,
               'Incumbent Market Share': i_market_share,
               'Incumbent Churn Ratio': i_churn_ratio,
               'Subscription': point1,
               'Churn': point2,
               'Market Share': point3,
               'ARPU': point4,
               'Incumbent Vulnerability': point5,
               'Total Points': point1+point2+point3+point4,
               'Point Legend': point_legend}
    cv = cv.append(cv_data, ignore_index=True)

# =======================================================================

# Read country map JSON file
json_input_file = 'countries.geo.json'
with open(json_input_file, 'r') as fr:
    world_map = fr.read()

# -----------------------------------------------------------------------

# Map OVUM countries to map countries
# MMissing countries
# Bahrain, Cabo Verde, Comoros, Cote d'Ivoire, French West Indies, Hong Kong
#Liechtenstein, Macao, Maldives, Mauritius, Monaco, Palestine, Réunion
# Sao Tome & Principe, Singapore, Timor-Leste

# Mappable countries
country_dict = {'Bosnia & Herzegovina': 'Bosnia & Herzegovina',
                'Congo': 'Democratic Republic of the Congo',
                'Guinea-Bissau': 'Guinea Bissau',
                'Serbia': 'Republic of Serbia',
                'Tanzania': 'United Republic of Tanzania',
                'USA': 'United States of America'}

# -----------------------------------------------------------------------

# Convert country names on the map if necessary
cv['Country on the map'] = cv['Country'].apply(
    lambda c: country_dict[c] if c in country_dict else c)

# Compile final outcome
cv['Result'] = cv['Last Follower'].apply(
    lambda c: 'Challenger: '+c if c != '' else ' ')
cv['Result'] += cv['Last Follower Market Share'].apply(
    lambda c: ' ('+"%.2f" % c+'%)' if not np.isnan(c) else ' ')
cv['Result'] += cv['Incumbent'].apply(
    lambda c: ', Incumbent: '+c if c != '' else ' ')
cv['Result'] += cv['Incumbent Market Share'].apply(
    lambda c: ' ('+"%.2f" % c+'%)' if not np.isnan(c) else ' ')

# Coloring
colors = ['darkblue']+['darkgreen']+['maroon']+['yellow']

# Calculate colors for the countries
cv['Color'] = cv['Total Points'].apply(lambda c: colors[int(c)])

# -----------------------------------------------------------------------

# Generate JSON output file
for index, row in cv.iterrows():
    # "name":"Angola" --> "name":"Angola","...":"...","...":"..."
    s_old = '"name":"'+row['Country on the map']+'"'

    s_new = s_old
    s_new += ',"Color":"'+row['Color']+'"'
    s_new += ',"Score":"'+str(int(row['Total Points']))+row['Point Legend']+'"'
    s_new += ',"Result":"'+row['Result']+'"'

    world_map = world_map.replace(s_old, s_new)

# -----------------------------------------------------------------------

# Handle countries that are on the map only
# {"name":"Montenegro","Color":"#7C0607","Score":"nan","Result":"    "}
# {"name":"Mongolia"}
i = 0
while i < len(world_map):
    # Search for country name first
    i = world_map.find('"name":"', i)
    if i >= 0:
        # Search for the second '"'
        j = world_map.find('"', i+8)+1
        if world_map[j] == '}':
            # Country on the map only
            world_map = world_map.replace(world_map[i:j],
                                          world_map[i:j]+',"Color":"'+colors[0]+'","Score":"0","Result":"'+chr(32)+'"', 1)
    if i < 0:
        break
    i = j+1

# -----------------------------------------------------------------------

# Save modified map
json_output_file = 'countries.geo.bsm.json'
with open(json_output_file, 'w') as fw:
    fw.write(world_map)

# =======================================================================

# Visualisation

# -----------------------------------------------------------------------

# Create list of all countries for all metrics
unique_full_country_list = list(mdf['Country'].sort_values().unique())

# -----------------------------------------------------------------------

# Get indexes for Subscriptions
#   countries['From'] & ['To'] contains the indexes for full countries (su_fcountries)
su_fcountries = pd.DataFrame(su_pivot.index.get_level_values(0))
su_indexes = pd.DataFrame(unique_full_country_list, columns=['Country'])
su_indexes['From'] = 0
su_indexes['To'] = 0
i = 0
for c in unique_full_country_list:
    try:
        a = su_fcountries[su_fcountries['Country'] == c].index[0]
        su_indexes.loc[[i], ['From']] = a
        su_indexes.loc[[i], ['To']] = a + \
            int(su_fcountries[su_fcountries['Country'] == c].count())
    # _=input('?')
    except:
        pass  # Do nothing in python
    i += 1

# -----------------------------------------------------------------------

# Get indexes for Churn
#   countries['From'] & ['To'] contains the indexes for full countries (ch_fcountries)
ch_fcountries = pd.DataFrame(ch_pivot.index.get_level_values(0))
ch_indexes = pd.DataFrame(unique_full_country_list, columns=['Country'])
ch_indexes['From'] = 0
ch_indexes['To'] = 0
i = 0
for c in unique_full_country_list:
    try:
        a = ch_fcountries[ch_fcountries['Country'] == c].index[0]
        ch_indexes.loc[[i], ['From']] = a
        ch_indexes.loc[[i], ['To']] = a + \
            int(ch_fcountries[ch_fcountries['Country'] == c].count())
    # _=input('?')
    except:
        pass  # Do nothing in python
    i += 1

# -----------------------------------------------------------------------

# Get indexes for Market Share
#   countries['From'] & ['To'] contains the indexes for full countries (ms_fcountries)
ms_fcountries = pd.DataFrame(ms_pivot.index.get_level_values(0))
ms_indexes = pd.DataFrame(unique_full_country_list, columns=['Country'])
ms_indexes['From'] = 0
ms_indexes['To'] = 0
i = 0
for c in unique_full_country_list:
    try:
        a = ms_fcountries[ms_fcountries['Country'] == c].index[0]
        ms_indexes.loc[[i], ['From']] = a
        ms_indexes.loc[[i], ['To']] = a + \
            int(ms_fcountries[ms_fcountries['Country'] == c].count())
    # _=input('?')
    except:
        pass  # Do nothing in python
    i += 1

# -----------------------------------------------------------------------

# Get indexes for ARPU
#   countries['From'] & ['To'] contains the indexes for full countries (su_fcountries)
ar_fcountries = pd.DataFrame(ar_pivot.index.get_level_values(0))
ar_indexes = pd.DataFrame(unique_full_country_list, columns=['Country'])
ar_indexes['From'] = 0
ar_indexes['To'] = 0
i = 0
for c in unique_full_country_list:
    try:
        a = ar_fcountries[ar_fcountries['Country'] == c].index[0]
        ar_indexes.loc[[i], ['From']] = a
        ar_indexes.loc[[i], ['To']] = a + \
            int(ar_fcountries[ar_fcountries['Country'] == c].count())
    # _=input('?')
    except:
        pass  # Do nothing in python
    i += 1

# -----------------------------------------------------------------------
# Create dropdown widget

OPENING_COUNTRY = 'Germany'
country_dropdown = Select(title='Countries:',
                          value=OPENING_COUNTRY,
                          options=unique_full_country_list)

# -----------------------------------------------------------------------


def prepare_chart(dataset,
                  chart_name,
                  country_indexes,
                  chart_title,
                  y_axis,
                  opening_country=OPENING_COUNTRY,
                  y_range=None,
                  divider=1):
    # Initialise
    ts_x_full_country_list_of_lists = []
    vals_y_full_country_list_of_lists = []
    full_country_colors_to_use = []
    # legend_to_use=[]
    full_country_list = []
    full_operator_list = []
    full_from_list = []
    full_to_list = []

    # Go for selected country
    # for x, row in ms_pivot.loc[country_dropdown.value,:].iterrows():
    i = 0
    for x, row in dataset.iterrows():
        date_list = [str(a.year)[-2:]+'Q'+str(int(a.month/3))
                     for a in list(dataset.columns)]  # Convert to quarters
        # xs determine number of lines on the chart
        ts_x_full_country_list_of_lists.append(date_list)
        vals_y_full_country_list_of_lists.append(row/divider)  # value
        full_country_list.append(x[0])  # country
        full_operator_list.append(x[1])  # operator
        full_country_colors_to_use.append(
            operators[operators['Operator'] == x[1]]['Color'])
        full_from_list.append(
            country_indexes[country_indexes['Country'] == x[0]]['From'].get_values()[0])
        full_to_list.append(
            country_indexes[country_indexes['Country'] == x[0]]['To'].get_values()[0])

        previous_country = x[0]
        i += 1

    # -----------------------------------------------------------------------

    # Prepare data with all country lines
    full_source = ColumnDataSource(data=dict(xs=ts_x_full_country_list_of_lists,
                                             ys=vals_y_full_country_list_of_lists,
                                             colors=full_country_colors_to_use,
                                             Country=full_country_list,
                                             Operator=full_operator_list,
                                             From=full_from_list,
                                             To=full_to_list))

    # -----------------------------------------------------------------------

    # Prepare selection data
    list_index = full_country_list.index(opening_country)
    from_item = full_from_list[list_index]
    to_item = full_to_list[list_index]

    ts_x_selected_country_list_of_lists = ts_x_full_country_list_of_lists[from_item:to_item]
    vals_y_selected_country_list_of_lists = vals_y_full_country_list_of_lists[
        from_item:to_item]
    selected_country_colors_to_use = full_country_colors_to_use[from_item:to_item]
    selected_country_alphas_to_use = [
        1 for a in range(len(selected_country_colors_to_use))]
    selected_operator_list = full_operator_list[from_item:to_item]

    if y_range == None:
        selected_p = figure(title=chart_title,
                            x_axis_label='Date',
                            y_axis_label=y_axis,
                            plot_width=SCM_CHART_WIDTH,
                            plot_height=SCM_CHART_HEIGHT,
                            x_range=date_list,
                            toolbar_location=None)
    else:
        selected_p = figure(title=chart_title,
                            x_axis_label='Date',
                            y_axis_label=y_axis,
                            plot_width=SCM_CHART_WIDTH,
                            plot_height=SCM_CHART_HEIGHT,
                            x_range=date_list,
                            toolbar_location=None,
                            y_range=y_range)  # Use common, linked y axis

    selected_source = ColumnDataSource(data=dict(xs=ts_x_selected_country_list_of_lists,
                                                 ys=vals_y_selected_country_list_of_lists,
                                                 colors=selected_country_colors_to_use,
                                                 alphas=selected_country_alphas_to_use,
                                                 Operator=selected_operator_list))

    selected_dataset_lines = selected_p.multi_line(xs='xs',
                                                   ys='ys',
                                                   source=selected_source,
                                                   line_color='colors',
                                                   line_alpha='alphas')

    # -----------------------------------------------------------------------

    selected_p.legend.location = 'top_left'

    selected_p.add_tools(HoverTool(renderers=[selected_dataset_lines],
                                   tooltips=[('Operator', '@Operator')]))

    # output to static HTML file
    output_file(sys.argv[0].replace('.py', '.html'))

    return selected_p, full_source, selected_source


su_chart, su_full_source, su_selected_source = prepare_chart(su_pivot, 'Subscriptions', su_indexes,
                                                             'OVUM WCIS - Subscriptions', 'Quarterly av. subscriptions (mio.)',
                                                             divider=1000000)
ch_chart, ch_full_source, ch_selected_source = prepare_chart(ch_pivot, 'Churn', ch_indexes,
                                                             'OVUM WCIS - Churn', 'Churn (%)')
ms_chart, ms_full_source, ms_selected_source = prepare_chart(ms_pivot, 'Market Share', ms_indexes,
                                                             'OVUM WCIS - Market Share', 'Market Share (%)')
ar_chart, ar_full_source, ar_selected_source = prepare_chart(ar_pivot, 'Market Share', ar_indexes,
                                                             'OVUM WCIS - ARPU', 'ARPU (Av. Revenue p. User)')

# -----------------------------------------------------------------------


def country_dropdown_callback(su_full_source=su_full_source,
                              su_selected_source=su_selected_source,
                              ch_full_source=ch_full_source,
                              ch_selected_source=ch_selected_source,
                              ms_full_source=ms_full_source,
                              ms_selected_source=ms_selected_source,
                              ar_full_source=ar_full_source,
                              ar_selected_source=ar_selected_source,
                              country_dropdown=country_dropdown,
                              window=None):

    # Nested function is problematic --> inline

    # Initialise data sources
    su_full = su_full_source.data
    su_selected = su_selected_source.data
    ch_full = ch_full_source.data
    ch_selected = ch_selected_source.data
    ms_full = ms_full_source.data
    ms_selected = ms_selected_source.data
    ar_full = ar_full_source.data
    ar_selected = ar_selected_source.data

    # Subscriptions
    i = 0
    len_countries = len(su_full.Country)
    while (i < len_countries) & (country_dropdown.value != su_full['Country'][i]):
        i += 1

    if i >= len_countries:
        su_selected.alphas = [0 for a in range(len(su_selected.xs))]
    else:
        su_list_index = su_full.Country.index(country_dropdown.value)
        su_from_item = su_full.From[su_list_index]
        su_to_item = su_full.To[su_list_index]

        # Determines number of lines!
        su_selected.xs = su_full.xs[su_from_item:su_to_item]
        su_selected.ys = su_full.ys[su_from_item:su_to_item]
        su_selected.colors = su_full.colors[su_from_item:su_to_item]
        su_selected.alphas = [1 for a in range(len(su_selected.xs))]
        su_selected.Operator = su_full.Operator[su_from_item:su_to_item]

    # Churn
    i = 0
    len_countries = len(ch_full.Country)
    while (i < len_countries) & (country_dropdown.value != ch_full['Country'][i]):
        i += 1

    if i >= len_countries:
        ch_selected.alphas = [0 for a in range(len(ch_selected.xs))]
    else:
        ch_list_index = ch_full.Country.index(country_dropdown.value)
        ch_from_item = ch_full.From[ch_list_index]
        ch_to_item = ch_full.To[ch_list_index]

        ch_selected.xs = ch_full.xs[ch_from_item:ch_to_item]
        ch_selected.ys = ch_full.ys[ch_from_item:ch_to_item]
        ch_selected.colors = ch_full.colors[ch_from_item:ch_to_item]
        ch_selected.alphas = [1 for a in range(len(ch_selected.xs))]
        ch_selected.Operator = ch_full.Operator[ch_from_item:ch_to_item]

    # Market Share
    i = 0
    len_countries = len(ms_full.Country)
    while (i < len_countries) & (country_dropdown.value != ms_full['Country'][i]):
        i += 1

    if i >= len_countries:
        ms_selected.alphas = [0 for a in range(len(ch_selected.xs))]
    else:
        ms_list_index = ms_full.Country.index(country_dropdown.value)
        ms_from_item = ms_full.From[ms_list_index]
        ms_to_item = ms_full.To[ms_list_index]

        # Determines number of lines!
        ms_selected.xs = ms_full.xs[ms_from_item:ms_to_item]
        ms_selected.ys = ms_full.ys[ms_from_item:ms_to_item]
        ms_selected.colors = ms_full.colors[ms_from_item:ms_to_item]
        ms_selected.alphas = [1 for a in range(len(ch_selected.xs))]
        ms_selected.Operator = ms_full.Operator[ms_from_item:ms_to_item]

    su_selected_source.change.emit()
    ch_selected_source.change.emit()
    ms_selected_source.change.emit()


country_dropdown.callback = CustomJS.from_py_func(country_dropdown_callback)
country_dropdown.width = DROPDOWN_WIDTH

# =======================================================================

# Compile bokeh world map
with open(r'countries.geo.bsm.json', 'r') as f:
    geo_source = GeoJSONDataSource(geojson=f.read())

#TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"
TOOLS = [BoxZoomTool(match_aspect=True), ResetTool(), HoverTool()]

# Full canvas of world countries
world_map_p = figure(width=1300,
                     height=750,
                     title='World Countries - Move mouse over country',
                     tools=TOOLS,
                     x_axis_label='Longitude',
                     y_axis_label='Latitude')

world_map_p.grid.grid_line_color = None

world_map_p.patches('xs', 'ys', fill_alpha=0.7, fill_color={'field': 'Color'},
                    line_color='black', line_width=0.5, source=geo_source)

world_map_p.background_fill_color = 'lightblue'
world_map_p.background_fill_alpha = 0.5

hover = world_map_p.select(dict(type=HoverTool))
hover.point_policy = 'follow_mouse'

hover.tooltips = """
    <table style="width:100%">
        <tr>
            <td><font face="Arial" size="3"><strong>@name</strong></td>
            <td><font face="Arial" size="3"><div align="right"><strong>Score: @Score</strong></div></td>
        </tr>
    </table>
    <font face="Arial" size="2">@Result</font>
"""

# -----------------------------------------------------------------------

output_file('BSM.html', title='BSM')

# -----------------------------------------------------------------------

# Put dropdown menu & chart in a column
layout = bokeh.layouts.column(bokeh.layouts.row(bokeh.layouts.column(bokeh.layouts.row(country_dropdown, chart_heading_plot),
                                                                     bokeh.layouts.row(text1_chart, su_chart, ch_chart, ms_chart))),
                              world_map_p)

# show the results
show(layout)
