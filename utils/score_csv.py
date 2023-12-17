import os
import re
import numpy as np
import pandas as pd




def get_metrics(ground_truth,generated_csv):
    intersected_list = []
    nonintersected_list = []
    gt_df=pd.read_csv(ground_truth,index_col=False)
    csv_df=pd.read_csv(generated_csv,index_col=False)
    #csv_df=pd.read_excel(generated_csv,index_col=False)

    has_equal_columns=len(gt_df.columns)==len(csv_df.columns)
    has_equal_rows=len(gt_df)==len(csv_df)

    gt_df=gt_df.replace(np.nan,0)
    gt_values=[]
    for col in gt_df.columns:
        gtcol=list(gt_df[col])
        gt_values.extend(gtcol)

    gt_values_=[str(x).replace(",","")
                .replace("(","")
                .replace(")","")
                .replace("$","")
                .replace(".","")
                .replace("-","")
                .strip()
                .lower() for x in gt_values if x !=np.nan]

    csv_df=csv_df.replace(np.nan,0)
    csv_cells=[]
    for col in csv_df.columns:
        csv_cells.extend(list(csv_df[col]))
    csv_cells_=[str(x).replace(",","")
                .replace("(","")
                .replace(")","")
                .replace("$","")
                .replace(".","")
                .replace("-","")
                .strip()
                .lower() for x in csv_cells if x !=np.nan]

    csv_cells_= [x.split('_')[-1] if x.startswith('row_') else x for x in csv_cells_]

    for g in set(gt_values_):
        if g in set(csv_cells_):
            intersected_list.append(g)
        else:
            nonintersected_list.append(g)
    print(f"{len(intersected_list)} cells values matched out of {len(set(gt_values_))}")
    recall=len(intersected_list)/(len(set(gt_values_)))
    metrics={"matched cell count":len(intersected_list),
             "GT cell count":len(set(gt_values_)),
             "extracted cell count":len(set(csv_cells_)),
             "union cell count":len(set(gt_values_))+len(set(csv_cells_)),
             "has equal rows": has_equal_rows,
             "has equal columns": has_equal_columns,
             "precision":len(intersected_list)/len(set(csv_cells_)),
             "recall":recall,
             "jaccard score":len(intersected_list)/(len(set(gt_values_))+len(set(csv_cells_)))
             }
    print(nonintersected_list)
    print(set(csv_cells_))
    return(metrics)


if __name__=="__main__":

    # gt_path="/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/"
    # myfiles=os.listdir(gt_path)
    # to_test=["/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/",
    #          "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp1/",
    #          "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/"
    #          ]
    # for mf in myfiles:
    #     metric_df =pd.DataFrame(columns=["Parameter","NMS-IOU","IDP1","IDP2"])
    #     metric_df["Parameter"]=['has equal rows','has equal columns','total no.of cells','total cells matched','accuracy']
    #     ground_truth=f"{gt_path}{mf}"
    #     for test in to_test:
    #         generated_csv=f"{test}{test.split('/')[-2]}_{mf}"
    #         metric=get_metrics(ground_truth,generated_csv)
    #         print(metric)
    ground_truth = "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/enersys_2.csv"
    # generated_csv = ("/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou"
    #                   "/enersys_webpage_3_1.csv")
    generated_csv = ("/home/suparna/PycharmProjects/TableExtraction/data/csv/idp2/idp2_enersys_webpage_2.csv")
    metric = get_metrics(ground_truth, generated_csv)
    print(metric)
    #gt=["/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/enersys_3.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Apple_34.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Apple_36.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Transdigm_131_4.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Transdigm_131_3.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Transdigm_131_2.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/Transdigm_96.csv",
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/enersys_1.csv " ,
    # "/home/suparna/PycharmProjects/TableExtraction/data/ground_truth/enersys_2.csv"]

    #nms_iou=["/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/enersys_webpage_3_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_Apple_34_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_Apple_36_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_4.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_3.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_2.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_96_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/IDP_compressed/Unbordered/Xerox_Table/output/xlsx_output/idp2_pytesseract_Apple_36.xlsx",
    #"/ home / suparna / PycharmProjects / TableExtraction / data / csv / nms_iou / nms_iou_enersys_webpage_1.csv",
    # "/home / suparna / PycharmProjects / TableExtraction / data / csv / nms_iou / nms_iou_enersys_webpage_2.csv"]

    #idp1=["/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp1/enersys_webpage_3_20.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp1/Apple_34_1_20.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp1/Apple_36_1_20.xlsx",
    # "/home/suparna/PycharmProjects/TableExtraction/data/csv/idp1/TransDigm_131_4.xlsx",
    # "/home/suparna/PycharmProjects/TableExtraction/data/csv/idp1/TransDigm_131_3.xlsx",
    # "/home/suparna/PycharmProjects/TableExtraction/data/csv/idp1/TransDigm_131_2.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp1/TransDigm_96_1_21.xlsx",
    # "/home/suparna/PycharmProjects/TableExtraction/data/csv/idp1/idp1_enersys_webpage_1.xlsx"
    # "/home/suparna/PycharmProjects/TableExtraction/data/csv/idp1/idp1_enersys_webpage_2.xlsx"]

    #idp2=["/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_enersys_webpage_3_1.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_Apple_34_1.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_Apple_36_1.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_TransDigm_131_4.xlsx"",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_TransDigm_131_3.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_TransDigm_131_2.xlsx",
    # "/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/idp2/idp2_TransDigm_96_1.xlsx",
    #"/home/suparna/PycharmProjects/TableExtraction/data/csv/idp2/idp2_enersys_webpage_1.csv",
    #"/home/suparna/PycharmProjects/TableExtraction/data/csv/idp2/idp2_enersys_webpage_2.csv" ]