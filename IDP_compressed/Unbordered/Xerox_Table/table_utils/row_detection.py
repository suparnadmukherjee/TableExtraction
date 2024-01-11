import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from IDP_compressed.Unbordered.Xerox_Table.table_utils import MinHVRunLengthCal as mhv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def FindLocalMinima(Data):
    DataInv = max(Data) - Data
    MinIdx, _ = find_peaks(DataInv)
    Minima = Data[MinIdx]
    return Minima, MinIdx


def SmoothingForLocalMinima(arr):
    dim = len(arr)
    newarr = arr
    for i in range(1, dim-1):
        avg = (arr[i-1] + arr[i] + arr[i+1])/3
        if avg < arr[i]:
            newarr[i] = avg
    return newarr


def row_detection(GryImg):
    (thresh, BwImg) = cv2.threshold(GryImg, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BwImgOrg = BwImg
    MinHVRLMat, MinHVRLModeVal, MinHVRLSecondModeVal, MinHVRLMinVal, MinHVRLMaxVal = mhv.MinHVRunLengthCal(BwImgOrg)

    if (MinHVRLModeVal != 0):
        FontThickness = MinHVRLModeVal
    else:
        FontThickness = MinHVRLSecondModeVal

    [ImSizeX, ImSizeY] = np.shape(BwImg)
    h = np.sum(BwImg, axis=1)
    Minima, MinIdx = FindLocalMinima(h)
    MinDistBtnLines = ImSizeX
    AvgDistBtnLines = ImSizeX / len(MinIdx)
    for i in range(1, len(MinIdx)):
        if (MinIdx[i] - MinIdx[i - 1] < MinDistBtnLines):
            MinDistBtnLines = MinIdx[i] - MinIdx[i - 1]

    while (MinDistBtnLines < 0.5 * AvgDistBtnLines):
        h = SmoothingForLocalMinima(h)
        Minima, MinIdx = FindLocalMinima(h)
        MinDistBtnLines = ImSizeX
        AvgDistBtnLines = ImSizeX / len(MinIdx)
        for i in range(1, len(MinIdx)):
            if MinIdx[i] - MinIdx[i - 1] < MinDistBtnLines:
                MinDistBtnLines = MinIdx[i] - MinIdx[i - 1]


    NoOfLines = len(MinIdx)
    BwImg[MinHVRLMat < 0.5 * FontThickness] = 1

    # DilateImg = ~imfill(~BwImg, 'holes');
    # ImgPerim = ~bwperim(~DilateImg);
    # DilateImg = ~xor(ImgPerim, DilateImg);

    se = np.ones((1, FontThickness), np.uint8)
    BWerode = cv2.erode(BwImg, se)
    ImgPerim = abs(BWerode - BwImg)
    BWerode = cv2.bitwise_xor(ImgPerim, BwImg)

    # se = strel('line', FontThickness, 0);
    # DilateImg = ~imdilate(~DilateImg, se)
    se = np.ones((1, FontThickness), np.uint8)
    ErodeImg = 1 - cv2.erode(1 - BWerode, se, iterations=1)
    DilateImg = 1 - cv2.dilate(1 - ErodeImg, se, iterations=1)

    # [LabelOfCC, NOCC] = bwlabel(~DilateImg);
    NOCC, LabelOfCC, stats, centroids = cv2.connectedComponentsWithStats(1 - DilateImg, 4, cv2.CV_32S)

    VRunLengthMat, VRunLengthMode, VRunLengthMax = mhv.PixVRLCal(DilateImg)
    OldDilateImg = DilateImg
    iteration = 0
    while iteration < 5 and NOCC > NoOfLines and (VRunLengthMax < 2 * (VRunLengthMode + 2) or VRunLengthMode < 3 * FontThickness):
        OldDilateImg = DilateImg
        # DilateImg = ~imdilate(~DilateImg, se);
        se = np.ones((1, FontThickness), np.uint8)
        DilateImg = 1 - cv2.dilate(1 - BwImg, se, iterations=1)
        # [LabelOfCC, NOCC] = bwlabel(~DilateImg);
        NOCC, LabelOfCC, stats, centroids = cv2.connectedComponentsWithStats(1 - DilateImg, 4, cv2.CV_32S)
        VRunLengthMat, VRunLengthMode, VRunLengthMax = mhv.PixVRLCal(DilateImg)
        iteration = iteration + 1
    DilateImg = OldDilateImg

    for i in range(NoOfLines):
        for j in range(2 * FontThickness):
            if MinIdx[i] + j < ImSizeX + 1:
                DilateImg[MinIdx[i] + j-1,:] = 0
    # cv2.imwrite('output/DilateImg_temp.jpg', DilateImg*255)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(1 - DilateImg, 4, cv2.CV_32S)
    line_mask = np.ones((np.shape(BwImg)[0],np.shape(BwImg)[1]), dtype="uint8")
    sep_line_mask = []
    old_y1 = 0
    for i in range(num_labels - 1):
        single_line_mask = np.ones((np.shape(BwImg)[0], np.shape(BwImg)[1]), dtype="uint8")
        area = stats[i + 1, cv2.CC_STAT_AREA]
        if area > FontThickness * ImSizeX:
            componentMask = (labels == i+1).astype("uint8")
            # cv2.imwrite('output/componentMask_' + str(i) + '.jpg', (1 - componentMask) * 255)

            x1 = stats[i + 1, cv2.CC_STAT_LEFT]
            y1 = stats[i + 1, cv2.CC_STAT_TOP] + stats[i + 1, cv2.CC_STAT_HEIGHT]
            x2 = stats[i + 1, cv2.CC_STAT_LEFT] + stats[i + 1, cv2.CC_STAT_WIDTH]
            y2 = stats[i + 1, cv2.CC_STAT_TOP]
            line_mask[int(y2):int(y1), int(x1):int(x2)] = 0
            # cv2.imwrite('output/line_mask_image_' + str(i) + '.jpg', line_mask * 255)
            if old_y1 > y2:
                line_mask[y2 + int((old_y1-y2)/2)-1 : y2 + int((old_y1-y2)/2)+1, int(x1):int(x2)] = 1
            old_y1 = y1
            single_line_mask[int(y2):int(y1), int(x1):int(x2)] = 0
            sep_line_mask.append(single_line_mask * 255)
            row_marked_image = cv2.rectangle(GryImg, (x1+5, y1), (x2-5, y2), (0, 0, 255), 1)
    # cv2.imwrite('output/row_marked_image.jpg', row_marked_image)
    # cv2.imwrite('output/line_mask_image.jpg', line_mask * 255)
    return line_mask * 255, sep_line_mask









