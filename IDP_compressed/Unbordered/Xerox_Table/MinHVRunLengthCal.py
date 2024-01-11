# %This function will calculate width of an image, where width of a image is
# %defined in the following way:
# %STEP 1: PixHRunLength = Calculate horizontal runlength of each pixel
# %STEP 2: PixVRunLength = Calculate vertical runlength of each pixel
# %STEP 3: PixelWidth = min(PixHRunLength,PixVRunLength)
# %STEP 4: ModeVal = highest frequency of PixelWidth
#
# %Sample:
#
# %*************************************************************
# %Developed by:                                               *
# %Chandan Biswas                                              *
# %Date: 06-09-2022                                            *
# %*************************************************************

import numpy as np
from scipy import stats as st


def PixVRLCal(InMat):
    [InMatSizX, InMatSizY] = np.shape(InMat)
    InMat[InMat > 0] = 1
    OutMat = np.zeros((InMatSizX, InMatSizY))
    for i in range(InMatSizX):
        for j in range(InMatSizY):
            if InMat[i, j] == 0:
                # Calculating vertical runlength of each pixel
                k = i
                PixVRunLength = 0
                while InMat[k, j] == 0:
                    PixVRunLength = PixVRunLength + 1
                    if k < InMatSizX - 1:
                        k = k+1
                    else:
                        break
                if i > 1:
                    k = i-1
                    while InMat[k, j] == 0:
                        PixVRunLength = PixVRunLength + 1
                        if k > 1:
                            k = k-1
                        else:
                            break
                # End of Calculation of vertical runlength of each pixel
                
                OutMat[i, j] = PixVRunLength
    MaxVal = max(OutMat.flatten())
    ModeVal = st.mode(OutMat.flatten())
    if int(ModeVal.mode[0]) == 0:
        newOutMat = np.delete(OutMat.flatten(), np.where(OutMat.flatten() == ModeVal))
        ModeVal = st.mode(newOutMat)
    return OutMat, int(ModeVal.mode[0]), MaxVal


def MinHVRunLengthCal(InMat):
    [InMatSizX, InMatSizY] = np.shape(InMat)
    InMat[InMat > 0] = 1
    OutMat = np.zeros((InMatSizX, InMatSizY))
    for i in range(InMatSizX):
        for j in range(InMatSizY):
            if InMat[i, j] == 0:
                # Calculating horizontal runlength of each pixel
                k = j
                PixHRunLength = 0
                while InMat[i, k] == 0:
                    PixHRunLength = PixHRunLength + 1
                    if k < InMatSizY - 1:
                        k = k + 1
                    else:
                        break
                if j > 1:
                    k = j - 1
                    while InMat[i, k] == 0:
                        PixHRunLength = PixHRunLength + 1
                        if k > 1:
                            k = k - 1
                        else:
                            break
                # End of Calculating horizontal runlength of each pixel

                # Calculating vertical runlength of each pixel
                k = i
                PixVRunLength = 0
                while InMat[k, j] == 0:
                    PixVRunLength = PixVRunLength + 1
                    if k < InMatSizX - 1:
                        k = k + 1
                    else:
                        break
                if i > 1:
                    k = i - 1
                    while InMat[k, j] == 0:
                        PixVRunLength = PixVRunLength + 1
                        if k > 1:
                            k = k - 1
                        else:
                            break
                # End of Calculating vertical runlength of each pixel

                OutMat[i, j] = min(PixHRunLength, PixVRunLength)

    MinVal = min(OutMat.flatten())
    MaxVal = max(OutMat.flatten())
    ModeVal = st.mode(OutMat.flatten())
    newOutMat = np.delete(OutMat.flatten(), np.where(OutMat.flatten() == ModeVal))
    SecondModeVal = st.mode(newOutMat)

    return OutMat, int(ModeVal.mode[0]), int(SecondModeVal.mode[0]), int(MinVal), int(MaxVal)
