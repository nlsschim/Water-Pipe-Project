import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as datetime

pipes_database = pd.read_csv('Tacoma_Pipe_Data.csv', low_memory=False)
breaks = pd.read_csv('Tacoma_Break_Data.csv')
print(pipes_database.shape)
print(breaks.shape)

pipes_database['Break_Yr'] = np.nan
pipes_database['Target'] = 0

a_list = []

for index, row in breaks.iterrows():
    # print(row)
    pipe_id = row['NEAR_FID']  # Location column
    if pipe_id != -1:
        real_pipe = pipes_database[pipes_database['OBJECTID'] == pipe_id]
        new_row = real_pipe.copy()
        new_row['Target'] = 1
        new_row['Break_Yr'] = row['Notif_date']
        #new_row['Width'] = row['Size']
        # new_row['TARGET_FID'] = -1
        a_list.append(new_row)

    # street_standard = (''.join([c for c in street if c.isalpha()])).lower()
    # for index, row in pipes_database.i:
    #     all_street = all_row['TARGET_FID'] # FULL_STRNA column
    #     # if all_street_standard in street_standard:
    #     #     possible_pipes = break_refs.setdefault(row, [])
    #     #     possible_pipes.append(all_row[1])
new_data = pipes_database.append(pd.concat(a_list))

print(pipes_database.shape)

new_data.to_csv('all_the_stuff.csv', header=True, index=False)
frederick = new_data


frederick = pd.read_csv('all_the_stuff.csv')
frederick.columns

frederick = frederick.rename(columns=
{
    'Join_Count_1': 'Breaks_nearby_2003',
    'Join_Count_12': 'Breaks_nearby_2004',
    'Join_Count_12_13': 'Breaks_nearby_2005',
    'Join_Count_12_13_14': 'Breaks_nearby_2006',
    'Join_Count_12_13_14_15': 'Breaks_nearby_2007',
    'Join_Count_12_13_14_15_16': 'Breaks_nearby_2008',
    'Join_Count_12_13_14_15_16_17': 'Breaks_nearby_2009',
    'Join_Count_12_13_14_15_16_17_18': 'Breaks_nearby_2010',
    'Join_Count_12_13_14_15_16_17_18_19': 'Breaks_nearby_2011',
    'Join_Count_12_13_14_15_16_17_18_19_20': 'Breaks_nearby_2012',
    'Join_Count_12_13_14_15_16_17_18_19_20_21': 'Breaks_nearby_2013',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22': 'Breaks_nearby_2014',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23': 'Breaks_nearby_2015',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23_24': 'Breaks_nearby_2016',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23_24_25': 'Breaks_nearby_2017',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26': 'Breaks_nearby_2018',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27': 'Breaks_nearby_2019',
    'Join_Count_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28': 'Breaks_nearby_2020',
    #'MNL_INSTAL': 'Install_year',
    #'MNL_MATE_1': 'Material',
    'Shape_Length': 'Length',
    #'SPEEDLIMIT': 'Speed_limit',
    #'SURFACETYP': 'Surface_type',
    'FID_ActualTacomaSoil': 'Soil_type',
    #'SURFACEWID': 'Surface_width',
    #'ARTCLASS': 'Arterial_class',
    'gridcode': 'Slope'

})

nearby_col_list = ['Breaks_nearby_2003',
        'Breaks_nearby_2004',
        'Breaks_nearby_2005',
        'Breaks_nearby_2006',
        'Breaks_nearby_2007',
        'Breaks_nearby_2008',
        'Breaks_nearby_2009',
        'Breaks_nearby_2010',
        'Breaks_nearby_2011',
        'Breaks_nearby_2012',
        'Breaks_nearby_2013',
        'Breaks_nearby_2014',
        'Breaks_nearby_2015',
        'Breaks_nearby_2016',
        'Breaks_nearby_2017',
        'Breaks_nearby_2018',
        'Breaks_nearby_2019',
        'Breaks_nearby_2020']



for col in nearby_col_list: #doesn't quite match our data
    print(col)
    print(frederick[col].sum())

columns = [
       'OBJECTID',
       'MATERIAL',
       'Length',
       'INSTALLDATE',
       'Soil_type',
       #'Arterial_class',
       #'Surface_width',
       #'Surface_type',
       #'Speed_limit',
       'Slope',
       'DIAMETER',
       'Breaks_nearby_2003',
       'Breaks_nearby_2004',
       'Breaks_nearby_2005',
       'Breaks_nearby_2006',
       'Breaks_nearby_2007',
       'Breaks_nearby_2008',
       'Breaks_nearby_2009', 'Breaks_nearby_2010',
       'Breaks_nearby_2011', 'Breaks_nearby_2012', 'Breaks_nearby_2013',
       'Breaks_nearby_2014', 'Breaks_nearby_2015', 'Breaks_nearby_2016',
       'Breaks_nearby_2017', 'Breaks_nearby_2018', 'Breaks_nearby_2019',
       'Breaks_nearby_2020',
       'Break_Yr',
       'T'
       ''
       'arget']
#print(frederick['Break_Yr'])
#frederick['INSTALLDATE'] = pd.to_datetime(frederick['INSTALLDATE'], format='%m/%d/%Y %H:%M:%S')
#frederick['INSTALLDATE'] = frederick['INSTALLDATE'].map(lambda x: x.year)

#frederick['Break_Yr'] = pd.to_datetime(frederick['Break_Yr'], format='%m/%d/%Y %H:%M:%S')
#frederick['Break_Yr'] = frederick['Break_Yr'].map(lambda x: x.year)

frederick['Soil_type'] = frederick['Soil_type'].map(lambda x: str(x))

frederick.tail()



frederick = frederick[columns]

#frederick['Width'] = pd.to_numeric(frederick['Width'], errors='coerce')

dummy_df = pd.get_dummies(frederick)
dummy_df.columns



dummy_df['Breaks_nearby_2009'] = dummy_df['Breaks_nearby_2009'].fillna(0).astype(int)



dummy_df.columns


def get_data(df, year):
    """
    Takes in df and filters depending on timeframe (start and end years).
    Returns subset of data for training in timeframe.
    """
    # want to include where there is NOT a break year (those will be our non-broken positive examples --> 'DATE' == 0) - DATE col

    # want to include where break year is in time frame of what we want - DATE col
    filt_df = df[(df['Break_Yr'].isnull()) | (df['Break_Yr'] == year)]

    # exclude installs AFTER time frame window - MNL_INSTAL col
    filt_df = filt_df[(filt_df['Install_year'] <= year)]

    # based on time window, calculate appropriate age of pipes (select beginning year of time frame --> ex: 2009, 2010, 2011
    #       subtract install year from 2009) - MNL_INSTAL col
    # -- will create negative numbers
    filt_df['AGE'] = year - filt_df['Install_year']

    filt_df['Nearby_breaks_1yr'] = filt_df[f"Breaks_nearby_{year - 1}"]
    print(filt_df[f"Breaks_nearby_{year - 1}"].sum())
    print(filt_df['Nearby_breaks_1yr'].sum())

    filt_df['Process_year'] = year

    for i in range(2003, 2021):
        filt_df = filt_df.drop(f"Breaks_nearby_{i}", axis=1)

    return filt_df


def main_function(dummy_df):
    """
    Takes in dummy dataframe and runs all functions based on given year ranges.
    Prints out year and accuracy per time range (3 years ranges)
    """
    output_list = []

    for process_year in range(2004, 2021):
        output_df = get_data(dummy_df, process_year)
        # num_breaks = output_df[output_df['Target'] == 1]
        # print(num_breaks.shape)
        # print()
        output_list.append(output_df)
        # print(file_name)
        # df = get_data(dummy_df, start_train, end_train)
        # break
    final_data = pd.concat(output_list)
    final_data.to_csv('final_data.csv', header=True, index=False)

final_data = pd.read_csv('final_data.csv')