# import the necessary packages

from PIL import Image
from numpy import asarray
import pandas as pd
import numpy as np
import cv2
import enchant
from sklearn.cluster import AgglomerativeClustering

from extraction.text_extraction import paddelExtract


csv_out_path="data/csv/"
def restructure_table(image_path,coords,ocrText):
    args = {
        "image": image_path,
        "output": f"args['image].split('/')[-1][:-3]_results.csv",
        "min_conf": 0,
        "dist_thresh": 30.0,
        "min_size": 1,
    }
    # set a seed for our random number generator
    np.random.seed(42)

    # load the input image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # load the image and convert into
    # numpy array
    img = Image.open(image_path)

    # asarray() class is used to convert
    # PIL images into NumPy arrays
    table = asarray(img)

    # <class 'numpy.ndarray'>
    #print(type(table))

    # shape
    #print(table.shape)
    d = enchant.Dict("en_US")


    xdiff = []
    ydiff = []
    for i, c in enumerate(coords[:-2]):
        i = i + 1
        # while i<len(coords)-1:
        # #print(i)
        xd = abs(coords[i][0] - coords[i + 1][0])
        yd = abs(coords[i][1] - coords[i + 1][1])
        xdiff.append([xd, ocrText[i], ocrText[i + 1]])
        ydiff.append([yd, ocrText[i], ocrText[i + 1]])
    xdiff=sorted(xdiff, key=lambda x: x[0])
    # for i in xdiff:
    #     #print(i)
    # for i in range(0,len(coords)):
        #print(coords[i],':',ocrText[i])

    # Extract all y-coordinates from the text bounding boxes, setting the x-coordinate value to zero
    #xCoords = [(c[0], 0) for c in coords]
    yCoords = [(0, c[1]) for c in coords]
    # Apply hierarchical agglomerative clustering to the coordinates
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="manhattan",
        linkage="complete",
        distance_threshold=args["dist_thresh"],
        compute_distances=True)
    clustering.fit(yCoords)
    #print(clustering.labels_)



    table_dict = {}
    clus_labels = np.unique(clustering.labels_)
    for l in clus_labels:
        table_dict[l] = []
        #print(f"{l}---------------------------------------------")
        indx = np.where(clustering.labels_ == l)

        # 1.create an array of the coordinates in a cluster
        cluster_coords = []
        for i in indx[0]:
            cluster_coords.append(coords[i])
        cc_arr = np.array(cluster_coords)
        #print("cc_arr", cc_arr)

        #print("2.sort all coords of the cluster by x axis:")
        cc_arr = cc_arr[np.argsort(cc_arr[:, 0])]
        cl_arr = cc_arr
        #print(cl_arr)

        #print("3. make subclusters")
        x_th = 10
        y_th = 10
        subclusters = []

        while len(cl_arr) != 0:
            newcluster = []
            newcluster.append(list(cl_arr[0]))
            #print("newcluster", newcluster)
            xs, ys, ws, hs = cl_arr[0]
            cl_arr = np.delete(cl_arr, 0, 0)
            #print("cl_arr", cl_arr)

            for c in cl_arr:
                s_node = newcluster[0]
                #print("c", c)
                #print("s_node", s_node)
                # #print(s_node[0])
                # #print(s_node[2])
                # #print(s_node[1])
                # #print(s_node[3])
                if (c[0] <= s_node[0] + s_node[2] + x_th) and (c[1] <= s_node[1] + s_node[3] + y_th):
                    newcluster.append(list(c))
                    #print("newcluster", newcluster)

                    i = np.where(cl_arr[:, :] == c)[0][0]
                    #print(i)
                    cl_arr = np.delete(cl_arr, i, 0)
            #print("done")
            #print(cl_arr)
            subclusters.append(newcluster)
            #print("subclusters", subclusters)

            #print("4.for each subcluster ")

            for sc in subclusters:
                #print("4a. sort by y coords")
                #print("sc", sc)
                sc.sort(key=lambda x: x[1])
                #print("sorted sc", sc)

                # #print("4b. make a copy of sorted sc")
                # sc_copy=sc[:]

                # #print("4c. Expand the coordinates")
                # for tup in sc[1:]:
                #     if tup[1]<=sc[0][1]+5:
                #         sc[0][2]=tup[0]+tup[2]-sc[0][0]
                #     if tup[1]-(sc[0][1]+sc[0][3])<=20:
                #         sc[0][3]=tup[1]+tup[3]-sc[0][1]
                # expanded_coordinates=tuple(sc[0])
                # #print("Expanded coordinates",tuple(sc[0]))

                sc = sorted(sc, key=lambda x: x[1], reverse=False)
                #print(sc)

                clus_text = ""
                # color = np.random.randint(0, 255, size=(3,), dtype="int")
                # color = [int(c) for c in color]
                # (x,y,w,h)=(803, 18, 112, 40)
                # cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)
                # clus_text=""
                for cc in sc:
                    clus_text += ocrText[coords.index(tuple(cc))]
                #print(clus_text)
            table_dict[l].append([sc, clus_text])

    #print(table_dict)

    regrouped_td={}
    for key,value in table_dict.items():
        #print("key",key)
        #print("value",value)
        a=value
        l_reclustered=[]
        x_th=8
        regrouped=[]
        col_coords=[x[0] for x in a]
        #print(col_coords)
        regrouped=col_coords[0]
        col_coords=col_coords[1:]
        #print("regrouped",regrouped)
        for i in range(0,len(col_coords)):
            ccd=col_coords[i]
            #print("ccd",ccd)
            #print(regrouped)
            if ccd[0][0]<=regrouped[-1][0]+regrouped[-1][2]+x_th:
                for c in ccd:
                    regrouped.append(c)
                    #print(regrouped,"group extended")
            else:
                l_reclustered.append(regrouped)
                regrouped=ccd
                #print(regrouped,"regrouped reinitialized")
        l_reclustered.append(regrouped)
        regrouped_td[key]=l_reclustered
    #print(regrouped_td)

    import regex as re

    text_extendedcoods_dict = {}
    all_coords = []
    for key, value in regrouped_td.items():
        text_extendedcoods_dict[key] = []
        #print(key)
        newvalue = []
        # #print(value)
        for v in value:
            extended_coods = []
            #print(v)
            text = ""
            for tc in v:
                text += ocrText[coords.index(tuple(tc))]

            #print(text)
            extended_coods.extend(v[0][:2])
            max_x = -1
            max_y = -1
            max_x_pos = -1
            max_y_pos = -1
            for i in range(0, len(v)):
                if v[i][0] > max_x:
                    max_x = v[i][0]
                    max_x_pos = i
                if v[i][1] > max_y:
                    max_y = v[i][1]
                    max_y_pos = i

            w = (v[max_x_pos][0] + v[max_x_pos][2]) - v[0][0]
            h = (v[max_y_pos][1] + v[max_y_pos][3]) - v[0][1]
            extended_coods.extend([w, h])
            #print(extended_coods)
            if re.search('[a-zA-Z]', text) == None and ('$' not in text) and ('-' not in text):
                text = ''.join(e for e in text if e.isalnum())
                # text=int(text.replace(',',''))
                #print(text)
            text_extendedcoods_dict[key].append([key, extended_coods, text])
            all_coords.append([key, extended_coods, text])
    #print(text_extendedcoods_dict)
    text_extendedcoods_dict =dict (sorted(text_extendedcoods_dict.items(), key=lambda x:x[1][0][1][1]))
    #print(text_extendedcoods_dict)
    text_extendedcoods_dict = {i: v for i, v in enumerate(text_extendedcoods_dict.values())}
    #print(text_extendedcoods_dict)
    for key,value in text_extendedcoods_dict.items():
        for v in value:
            v[0]=key
    #print(text_extendedcoods_dict)
    all_coordslist=(text_extendedcoods_dict.values())
    #print(all_coordslist)
    coords_list=[ci  for vi in all_coordslist for ci in vi ]
    #print(coords_list)
    coords_list=sorted(coords_list, key=lambda index : index[1][0])
    #print(coords_list)
    columns = []
    coords_list_bk = coords_list

    col_i = []
    # #print(col_i)
    while len(coords_list_bk) > 0:
        if len(col_i) == 0:
            #print("newcolumn")
            col_i = [''] * len(clustering.labels_)
            # #print(coords_list_bk[0])
            row = coords_list_bk[0][0]
            # #print(row)
            col_i[row] = coords_list_bk[0]
            #print(col_i)
            coords_list_bk = coords_list_bk[1:]
            # #print(col_i)
            r = col_i[row][1][0]
            q1 = (r + col_i[row][1][2]) / 4
            m = (r + col_i[row][1][2]) / 2
            q3 = 3 * (r + col_i[row][1][2]) / 4
            l = (r + col_i[row][1][2])


        else:
            row = coords_list_bk[0][0]
            # #print(coords_list_bk[0])
            xmin, xmax = int(coords_list_bk[0][1][0]), int(coords_list_bk[0][1][0] + coords_list_bk[0][1][2])
            # #print(xmax,"-",xmin)
            if r in range(xmin, xmax + 1) or q1 in range(xmin, xmax + 1) or m in range(xmin, xmax + 1) or q3 in range(xmin,
                                                                                                                      xmax + 1) or l in range(
                    xmin, xmax + 1):
                col_i[row] = coords_list_bk[0]
                #print(col_i)
                coords_list_bk = coords_list_bk[1:]


            else:

                #print(col_i)
                columns.append(col_i)
                col_i = []
    columns.append(col_i)

    #print(columns)
    df = pd.DataFrame(columns)
    df=df.T
    #print(df)
    for col in df:
        df[col]=df[col].apply(lambda x: x[2] if x!='' else x)

    #if exel file required
    #excel_file=image_path.split('/')[-1][:-4]
    #print(excel_file)
    #df.to_excel(f"{excel_file}.xlsx")

    #write to csv
    csvfname=(f"{csv_out_path}/mhac/mhac_{image_path.split('/')[-1]}.csv")
    print(csvfname)
    df.to_csv(f"{csvfname}.csv")

def get_table(image_path):
    output=paddelExtract(image_path)
    coords=[]
    ocrText=[]
    for i in range(0, len(output)):
        x = output[i][0][0][0]
        y = output[i][0][0][1]
        w = output[i][0][2][0] - x
        h = output[i][0][2][1] - y
        text = output[i][1][0]
        coords.append((x, y, w, h))
        ocrText.append(text)
    restructure_table(image_path,coords,ocrText)

if __name__=="__main__":
    get_table("/home/suparna/PycharmProjects/TableDetection/Approaches_results/Apple/testImages/Apple__36_cropped_margin10_1.png")