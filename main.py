import os

import pandas as pd
from numpy import NaN


def main():
    res = pd.read_csv("files/main_task.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res.info())


if __name__ == '__main__':
    main()
