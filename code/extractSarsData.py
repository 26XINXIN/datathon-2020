import csv
import sys, os
import numpy as np

def read_data():
    for filename in os.listdir("../data/sars"):
        m, d, y = filename.split('.')[0].split('_')
        date = y + '-' + m + '-' + d
        data = list()
        with open(filename) as f:
            reader = csv.reader(f)
            for line in enumerate(reader):
                if i == 0:
                    header = line
                else:
                    