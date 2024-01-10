import os
import pandas as pd
import html5lib
ground_truth_path=f"/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/"
def html_str_to_df(tablestr,file_name):

    dfs = pd.read_html(tablestr)
    df = dfs[0]
    df=df.fillna("")

    csvname=f"{ground_truth_path}{file_name}.csv"
    df.to_csv(csvname)
    print(df)

if __name__=="__main__":

    fname="/home/suparna/PycharmProjects/TableExtraction/data/htmltabletxt/Apple_36.txt"
    file_name = os.path.basename(fname)
    with open("/home/suparna/PycharmProjects/TableExtraction/data/htmltabletxt/Apple_36.txt",'r') as f:
        tablestr=f.readlines()
    html_str_to_df(tablestr[0],file_name)
    #"/home/suparna/PycharmProjects/TableExtraction/data/htmltabletxt/enersys_webpage_1.txt"
    #"/home/suparna/PycharmProjects/TableExtraction/data/htmltabletxt/enersys_webpage_2.txt"