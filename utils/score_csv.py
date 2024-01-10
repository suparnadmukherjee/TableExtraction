import os
import re
import numpy as np
import pandas as pd




def get_metrics(ground_truth,generated_csv):
    intersected_list = []
    nonintersected_list = []

    gt_df=pd.read_csv(ground_truth,dtype=str)
    if generated_csv[-3:]=="csv":
        csv_df=pd.read_csv(generated_csv,dtype=str)
    else:
        csv_df=pd.read_excel(generated_csv,index_col=False)

    has_equal_columns=len(gt_df.columns)==len(csv_df.columns)
    has_equal_rows=len(gt_df)==len(csv_df)

    gt_df=gt_df.replace(np.nan,0.0)
    gt_df = gt_df.replace('$', '')
    gt_vals=[]
    for col in gt_df.columns:
        gtcol=list(gt_df[col])
        gt_vals.extend(gtcol)
    # for i in range(0,len(gt_values)):
    #     if isinstance(gt_values[i], float):
    #         print(gt_values[i])
    #         gt_values[i]=int(gt_values[i])
    #         print(gt_values[i])
    gt_values=[i for i in gt_vals if i != 0.0]
    gt_values_=[str(x).replace(",","")
                .replace("(","")
                .replace(")","")
                .replace("$","")
                .replace(".","")
                .replace("-","")
                .strip()
                .lower() for x in gt_values if x !=np.nan]

    csv_df=csv_df.replace(np.nan,0.0)
    csv_df=csv_df.replace('$','')
    csv_vals=[]
    for col in csv_df.columns:
        csv_vals.extend(list(csv_df[col]))
    # for i in range(0, len(csv_cells)):
    #     if isinstance(csv_cells[i], float):
    #         print(csv_cells[i])
    #         csv_cells[i] = int(csv_cells[i])
    #         print(csv_cells[i])
    csv_cells = [i for i in csv_vals if i != 0.0]
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
    metrics={"filename":os.path.basename(generated_csv),
             "matched cell count":len(intersected_list),
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
    ground_truth = '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/enersys_webpage_3_1.csv'
    # generated_csv = ("/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou"
    #                   "/enersys_webpage_3_1.csv")
    generated_csv = ("/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms/nms_iou_enersys_webpage_3_1.csv")
    metric = get_metrics(ground_truth, generated_csv)
    print(metric)


    # annotation_gt=['/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/TransDigm_131_2.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Amazon_52.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/TransDigm_131_3.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/enersys_webpage_1_1.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Tesla_69.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/TransDigm_131_1.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Amazon_30.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Tesla_66.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/TransDigm_96_1.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Amazon_44.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Amazon_55.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/enersys_webpage_3_1.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Apple_36.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Tesla_71.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/enersys_webpage_2_1.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Apple_34.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/TransDigm_131_4.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Tesla_77.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Amazon_61.csv',
    #  '/home/suparna/PycharmProjects/TableExtraction/data/table_groundtruth/Apple_24.csv']

    #nms_iou=["/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/enersys_webpage_3_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_Apple_34_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_Apple_36_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_4.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_3.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_131_2.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_TransDigm_96_1.csv",
    #/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/nms_iou_enersys_webpage_1.csv",
    #"/home/suparna/PycharmProjects/TableDetection/TableExtractionDataBackup/csv/nms_iou/enersys_webpage_2_1.csv",
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

