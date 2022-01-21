'''
mAP result for testdataset, mitochodnria

'''

import numpy as np
import sys, os 
import mrcfile
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import tifffile
from scipy import ndimage as ndi
import skimage
from skimage import morphology
import random
import copy
import mrcfile
import pandas as pd
  

def read_isgfile(idx):
    path1 = f'F:/salilab/salilab/projects/organelle-separation/benchmark'
    # path2 = f'F:/salilab/salilab/projects/organelle-separation/analysis_data/1201_mito_output_fake' 
    path2 = f'F:/salilab/salilab/projects/organelle-separation/analysis_data/1201_mito_output_fake_watershed'  #newmito_output_fake_watershed'

    # idxlst = [i for i in range(10)]

    filename1 = f'{path1}/mito_sample_5mito_{idx}.tiff'
    # filename2 = f'{path2}/{idx}_instance_mask.tiff'
    filename2 = f'{path2}/{idx}_watershed_mask.tiff'
     
    return tifffile.imread(filename1), tifffile.imread(filename2)

def resort_idx(img):
    indexlst = list(set(list(img.reshape(1,-1))[0]))
    indexlst.remove(0)
    for i, idx in enumerate(indexlst):
        img[np.where(img == idx)] = i+1

    return img


class set_AP:
    def __init__(self, groundtruth_tiff, seg_tiff) -> None:
        self.groundtruth_tiff = groundtruth_tiff
        self.seg_tiff = seg_tiff
        self.accurcy_for_eachlabel = []

    def get_AP(self, threashold_acc = 0.5):

        label_in_gth = list(set(list(self.groundtruth_tiff.reshape(1,-1))[0]))
        label_in_gth.remove(0)

        label_in_seg = list(set(list(self.seg_tiff.reshape(1,-1))[0]))
        label_in_seg.remove(0)


        matched_label_lst = []
        for label in label_in_gth:
            # print(self.seg_tiff[np.where(self.groundtruth_tiff == label)])
            temp_lst = list(set(list(self.seg_tiff[np.where(self.groundtruth_tiff == label)].astype(int))))

            for idx2 in temp_lst:   
                matched_label_lst.append([idx2, int(label) ]) #[seglabel, gth,] 

        matched_label_lst_confidence1 = []
        matched_label_lst_confidence2 = []
        for pair in matched_label_lst:   #[seglabel, gth] 
            matched_label_lst_confidence1.append(len(np.array(np.where(self.seg_tiff == pair[0]))[0]))
            matched_label_lst_confidence2.append(len(np.array(np.where(self.seg_tiff == pair[1]))[0]))

        for i in range(len(matched_label_lst)):
            for j in range(i, len(matched_label_lst)):
                if matched_label_lst_confidence1[j] > matched_label_lst_confidence1[i]:
                    matched_label_lst[j], matched_label_lst[i] = matched_label_lst[i], matched_label_lst[j]
                    matched_label_lst_confidence1[j], matched_label_lst_confidence1[i] = matched_label_lst_confidence1[i], matched_label_lst_confidence1[j]
                    matched_label_lst_confidence2[j], matched_label_lst_confidence2[i] = matched_label_lst_confidence2[i], matched_label_lst_confidence2[j]

        for i in range(len(matched_label_lst)):
            for j in range(i, len(matched_label_lst)):
                if matched_label_lst_confidence1[j] == matched_label_lst_confidence1[i]:
                    if matched_label_lst_confidence2[j] > matched_label_lst_confidence2[i]:
                        matched_label_lst[j], matched_label_lst[i] = matched_label_lst[i], matched_label_lst[j]
                        matched_label_lst_confidence1[j], matched_label_lst_confidence1[i] = matched_label_lst_confidence1[i], matched_label_lst_confidence1[j]
                        matched_label_lst_confidence2[j], matched_label_lst_confidence2[i] = matched_label_lst_confidence2[i], matched_label_lst_confidence2[j]


        # print('line50',matched_label_lst)

        acc_TP = 0
        acc_FP = 0
        precision_lst = []
        recall_lst = []
        accuracy_dataframe = pd.DataFrame(columns=[i for i in range(len(label_in_gth))])  # col is gth


        for i, seglabel in enumerate(label_in_seg):
            for j, label in enumerate(label_in_gth):
                overlap = list(self.seg_tiff[np.where(self.groundtruth_tiff == label)].astype(int)).count(seglabel)
                overall = len(np.where(self.groundtruth_tiff == label)[0]) + len(np.where(self.seg_tiff == seglabel)[0]) - overlap
                accuracy = overlap / overall 
                accuracy_dataframe.loc[i,j] = accuracy

        # print(accuracy_dataframe)


        for pair in matched_label_lst: # pair = [seglabel, gth,] 
            # accuracy
            accuracy = accuracy_dataframe.loc[pair[0]-1, pair[1]-1]
            # print(accuracy_dataframe.loc[pair[0]-1,:], np.max(accuracy_dataframe.loc[pair[0]-1,:]))

            if accuracy >= np.max(accuracy_dataframe.loc[pair[0]-1,:]):
                if accuracy > threashold_acc:
                    self.accurcy_for_eachlabel.append(accuracy)
                    acc_TP += 1
                else:
                    acc_FP += 1
            
                precision_lst.append(acc_TP / (acc_TP + acc_FP))
                recall_lst.append(acc_TP / len(label_in_seg) )



        # print('line83', precision_lst, recall_lst)

        mAP_x = recall_lst
        mAP_y = [ max(precision_lst[i:] ) for i in range(len(precision_lst)) ]

        # plt.plot(mAP_x, mAP_y)
        # plt.show()
        # plt.close()

        return np.average(mAP_y)

    # def get_AP_50(self):
    #     return get_AP(self, 0.5)

    # def get_AP_70(self):
    #     return get_AP(0.7)

    # def get_AP_90(self):
    #     return get_AP(0.9)



def main():

    # idx = 0

    recalllst = np.arange(0.5,1,0.05)
    APlst = [[] for _ in range(len(recalllst))]
    for idx in range(10):
        print('image index',idx)
        gth_img, seg_img = read_isgfile(idx)
        seg_img = resort_idx(seg_img)

        # for i in range(gth_img.shape[0]):
        #     plt.imshow(gth_img[i])
        #     plt.show()
        #     plt.close()

        #     plt.imshow(seg_img[i])
        #     plt.show()
        #     plt.close()

        data1 = set_AP(gth_img, seg_img)
        #AP = AP[.50:.05:.95]
        

        for i in range(len(recalllst)): 
            APlst[i].append(data1.get_AP(recalllst[i]))
        # AP50 = data1.get_AP(0.5)
        # AP60 = data1.get_AP(0.6)
        # AP70 = data1.get_AP(0.7)
        # AP80 = data1.get_AP(0.8)  
        # AP90 = data1.get_AP(0.9)  
        
        # mAPlst[0].append(AP50)
        # mAPlst[1].append(AP60)
        # mAPlst[2].append(AP70)
        # mAPlst[3].append(AP80)
        # mAPlst[4].append(AP90)

    for i in range(len(APlst)):
        print(np.average(APlst[i]))

    print('mAP', np.average(APlst))
    print('AP50', np.average(APlst[0]))
    print('AP70', np.average(APlst[4]))
    print('AP90', np.average(APlst[8]))

        # print(mAP50, mAP60, mAP70, mAP80, mAP90)
        # print(data1.accurcy_for_eachlabel)

if __name__ == '__main__':
    main()  