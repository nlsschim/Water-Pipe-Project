import csv
import pandas as pd



"""
Needs full data file, break data file, install year column name, address column name, break year column name,
break location column name 
-- use group_by_id function only on kirkland data (outside of function)
"""

def search_allbreak_kirk(all_data, break_data, install_col, address_col, break_yr_col, break_loc):
    #searches for in kirk all pipe file for all matches from break file based on address

    all_df = group_by_id(readfile(all_data))
    break_df = readfile(break_data)

    #normalizes INSTYEAR and FULL_STRNA to make matching easier
    all_df[install_col]= all_df[install_col].str[4:8] #change INSTYEAR to just Year
    all_df[address_col] = [normalize(x) for x in all_df[address_col]]


    break_yr_filter = break_df[break_yr_col].tolist()
    break_yr_filter = list(map(str, break_yr_filter))
    break_st_filter = [remove_house_number(x) for x in break_df[break_loc]]
    # kirk_filter_year = kirk_all.INSTYEAR.isin(breakyear_filter)
    filter_street = all_df[address_col].isin(break_st_filter)

    # print(filter_street.head())
    filter_all = all_df[filter_street & (all_df[address_col] != "")]
    # print(filter_all.head())

    # uncomment to export to csv
    filter_all.to_csv(r'Possible_Matched_Pipes', index=None, header=True)

def split_addy(all_data, street_col):
    """
    For Seattle data - takes in street name and splits street names on "and".
    Creates two new columns of "street_one" and "street_two", and returns
    new dataframe.
    """
    street_one = []
    street_two = []
    for street_pair in all_data[street_col]:
        # if/else checks if there are two streets or just one
        if ' AND ' in street_pair:
            pair_array = street_pair.split(' AND ')
            street_one.append(pair_array[0])
            street_two.append(pair_array[1])
        else:
            street_one.append(street_pair)
            street_two.append(street_pair)
    all_data['street_one'] = street_one
    all_data['street_two'] = street_two
    return all_data


def search(street, all_data, street_col):
    """
    Takes an address from break data and returns/prints/exports dataframe
    associated with potential matches from all pipe dataset.
    """
    # for kirk data only
    # all_df = group_by_id(readfile(all_data))

    # For seattle data only
    all_df = split_addy(all_data, street_col)

    all_df[street_col] = [normalize(x) for x in all_df[street_col]]
    filtered = all_df[(all_df[street_col].str.match(remove_house_number(street)))]
    # print(filtered)

    filtered.to_csv(r'Potential_matches_{}'.format(street), index=None, header=True)
    print("SUCCESS!")


def normalize(str):
    return "".join(str.split()).lower()

def remove_house_number(str):
    #removes house number from string, potentially could be used to find intersections
    street = ""
    house_number = ""
    for string in str.split():
        if string.isdigit():
            house_number += string
        else:
            street += string
    return normalize(street)

def write_to_csv(df):
    #writes the DataFrame to CSV
    df.to_csv(r'Potential_Pipe_Match', index=None, header=True)

def group_by_id(df):
    grouping = df.groupby('FID_WA_Mai').groups
    unique_pipes = df['FID_WA_Mai'].unique()
    indices = []
    for i in range(len(unique_pipes)):
        pipe = unique_pipes[i]
        val = grouping[pipe][0]
        indices.append(val)

    # CALCULATE AVERAGE SLOPES FOR PIPES THAT GOT SPLIT INTO MULTIPLE ROWS
    slopes = df[['FID_WA_Mai', 'SLP_CLASS']]
    slopes = slopes.groupby('FID_WA_Mai')['SLP_CLASS'].mean()
    slopes = pd.Series(slopes)

    filtered = df.loc[indices, 'FID_WA_Mai':'FUNC_CLASS']
    filtered = filtered.drop('SLP_CLASS', axis=1)
    filtered['SLOPE'] = slopes.values

    return filtered


def readfile(filename):
    #reads file and returns a DataFrame
    df = pd.read_csv(filename)
    return df


def find_best_match():
    """
    Takes in potential matches csv file, and compares pipe size.
    Returns best matching pipe information.
    -- Info needed from all data set: material, install year, surface type, slope, length
    """


if __name__ == "__main__":
    #print(remove_house_number("724 14th AVE NE"))
    #search_bystreet_kirk()
    #search("696 16th Ave W")
    # search("696 16th Ave W", "seattle_data.csv", "INTRLO")
    # search_allbreak_kirk("Kirk_All.csv", "Kirk_Break.csv", "INSTYEAR", "FULL_STRNA", 
    #                       "Break_Year", "Location")
    seattle_df = readfile("seattle_data.csv")
    search("9619 24TH AVE NW", seattle_df, "INTRLO")