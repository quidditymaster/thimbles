
import numpy as np
import scipy
import thimbles as tmb
import scipy.stats
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--star-file", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    
    stardat = pd.read_csv("likely_members.csv")
    
