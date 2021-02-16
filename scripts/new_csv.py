import numpy as np
import pandas as pd

kirk_all = pd.read_csv('Kirk_All.csv', header=0)
kirk_break = pd.read_csv('Kirk_Break_Data.csv', header=0)
break_refs = {}
for index, row in kirk_break.iterrows():
    print(row)
    street = row[1] # Location column
    print(street)
    street_standard = (''.join([c for c in street if c.isalpha()])).lower()
    for all_row in kirk_all:
        print(all_row)
        all_street = all_row[10] # FULL_STRNA column
        all_street_standard = (''.join([c for c in all_street if c.isalpha()])).lower()
        if all_street_standard in street_standard:
            possible_pipes = break_refs.setdefault(row, [])
            possible_pipes.append(all_row[1])
        

print(break_refs)
        