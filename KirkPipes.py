import csv
import pandas as pd




def search_allbreak_kirk():
    #searches for in kirk all pipe file for all matches from break file based on address

    kirk_all = group_by_id(readfile("kirk_all.csv"))
    kirk_break = readfile("kirk_break.csv")

    #normalizes INSTYEAR and FULL_STRNA to make matching easier
    kirk_all["INSTYEAR"]=kirk_all["INSTYEAR"].str[4:8] #change INSTYEAR to just Year
    kirk_all["FULL_STRNA"] = [normalize(x) for x in kirk_all["FULL_STRNA"]]


    breakyear_filter = kirk_break["Break_Year"].tolist()
    breakyear_filter = list(map(str, breakyear_filter))
    breakstreet_filter = [remove_house_number(x) for x in kirk_break["Location"]]
    kirk_filter_year = kirk_all.INSTYEAR.isin(breakyear_filter)
    kirk_filter_street = kirk_all.FULL_STRNA.isin(breakstreet_filter)
    filter_all = kirk_all[kirk_filter_street & (kirk_all.FULL_STRNA != "")]

    

    #uncomment to export to csv
    #filter_all.to_csv(r'Possible_Matched_Pipes', index=None, header=True)

def search(street):
    #Takes an address and returns/prints/exports DataFrame Associated with potential matches
    kirk_all = group_by_id(readfile("kirk_all.csv"))
    kirk_break = readfile("kirk_break.csv")
    kirk_all["FULL_STRNA"] = [normalize(x) for x in kirk_all["FULL_STRNA"]]
    filtered = kirk_all[(kirk_all.FULL_STRNA.str.match(remove_house_number(street)))]
    print(filtered)

    #filtered.to_csv(r'Potential_matches_{}'.format(street), index=None, header=True)


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


if __name__ == "__main__":
    #print(remove_house_number("724 14th AVE NE"))
    #search_bystreet_kirk()
    #search("696 16th Ave W")
    search_allbreak_kirk()
