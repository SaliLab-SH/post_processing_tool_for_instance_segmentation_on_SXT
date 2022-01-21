# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot
import random
from scipy.spatial.distance import cdist
import sys, os
import pandas as pd
import copy
import matplotlib.pyplot as plt

from scipy.spatial.kdtree import distance_matrix

dirpath = os.path.dirname(__file__)
os.chdir(dirpath)
sys.path.append(os.pardir)
print('curpath', os.getcwd)



def read_testfile(check_ = 0):
    datapath = f'E:/PBC_data/scripts/granule-separation/test_datasets'
    mask = tifffile.imread(f'{datapath}/822_4_test_mito_mask.tiff')

    if check_:
        plt.imshow(mask[8])
        plt.show()
        plt.close()

    mrc = tifffile.imread(f'{datapath}/822_4_test_mito_mrc.tiff')
    # mrc = mrcfile.open(f'{datapath}/822_4_test_mrc.tiff', permissive=True).data
    if check_:
        plt.imshow(mrc[8])
        plt.show()
        plt.close()


    # for i in range(mask.shape[0]):
    #     fig = plt.figure()
    #     ax = plt.subplot(121)
    #     ax.imshow(mask[i])
    #     ax.set_title(f'mask i = {i}')

    #     ax = plt.subplot(122)
    #     ax.imshow(mrc[i])
    #     ax.set_title(f'mrc i = {i}')

    #     plt.show()
    #     plt.close()



    return mask, mrc





class K_Means(object):
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter


    def fit(self, data):
        self.centers_ = {}
        self.data = data
        for i in range(self.k_):
            self.centers_[i] = data[i]
        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            for feature in data:
                distances = []
                for center in self.centers_:
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)


            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break
    
    def closedist2(self):
        min_dist_lst = []
        for center in list(self.centers_.values()):
            dist_mat = cdist( np.array([center]), np.array(self.data)) [0]
            dist_mat = sorted(dist_mat)
            min_dist_lst.extend(dist_mat[:2])
        # print(min_dist_lst)

        return np.average(min_dist_lst)


    def elbowdist(self):
        dist_sqrure_lst = []
        for center in list(self.centers_.values()):
            dist_mat = cdist( np.array([center]), np.array(self.data)) [0]
            dist_mat_square = dist_mat * dist_mat
            dist_sqrure_lst.extend(np.sum(dist_mat_square))


    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index




def find_closest2vect(point, centerlst):
    temp_centlst = copy.deepcopy(centerlst)
    distlst = cdist(np.array([point]), np.array(centerlst))[0]

    for i in range(3):
        for j in range(i, len(distlst)):
            if distlst[i] > distlst[j]:
                distlst[i], distlst[j] = distlst[j], distlst[i]
                temp_centlst[i], temp_centlst[j] = temp_centlst[j], temp_centlst[i]

    vect1 = np.array(temp_centlst[1]) - np.array(temp_centlst[0])
    vect2 = np.array(temp_centlst[2]) - np.array(temp_centlst[0])

    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = vect2 / np.linalg.norm(vect2)

    return [temp_centlst[1], temp_centlst[2]], [vect1, vect2]


def find_vect(point, centerlst):
    temp_centlst = copy.deepcopy(centerlst)
    distlst = cdist(np.array([point]), np.array(centerlst))[0]

    for i in range(3):
        for j in range(i, len(distlst)):
            if distlst[i] > distlst[j]:
                distlst[i], distlst[j] = distlst[j], distlst[i]
                temp_centlst[i], temp_centlst[j] = temp_centlst[j], temp_centlst[i]

    vect1 = np.array(temp_centlst[1]) - np.array(temp_centlst[0])
    vect2 = np.array(temp_centlst[2]) - np.array(temp_centlst[0])

    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = vect2 / np.linalg.norm(vect2)

    # cos 30' =0.866
    length = abs(vect1 * vect2)
    threshold = 0.866

    if length > threshold:
        return list((vect1 + vect2) / 2)
    else:
        return []



class blob_classification:
    def __init__(self, mask, mrc) -> None:
        self.mrc = mrc
        self.mask = mask


    def fit(self, centerlst, radlst, check_ = 1):
        self.centerlst = centerlst
        self.radlst = radlst
        coordinfo = pd.DataFrame(columns=['coordinate','coord_idx','close2coord_idx','close2coord_cos_abs','average_vect', 'radius'])
        self.centerlabel = [0] * len(self.centerlst)

        coord_idx = [ i for i in range(len(centerlst))]

        cur_screen_idx = coord_idx  ## initial
        cur_screen_idx2 = copy.deepcopy(cur_screen_idx)
        labeled_idx1 = []
        dot_threshold = 0.866  #cos 45 = 0.707  # cos 30' =0.866   cos 10 = 0.984   cos15 = 0.965 cos20 = 0.939


        if len(centerlst) >= 3 :


            for idx, coord in enumerate(centerlst):  ## create dataframe to record center coords info
                
                temp_centlst = copy.deepcopy(centerlst)
                distlst = cdist(np.array([coord]), np.array(temp_centlst))[0]
                temp_idx = copy.deepcopy(coord_idx)

                for i in range(3):
                    for j in range(i, len(distlst)):
                        if distlst[i] > distlst[j]:
                            distlst[i], distlst[j] = distlst[j], distlst[i]
                            temp_centlst[i], temp_centlst[j] = temp_centlst[j], temp_centlst[i]
                            temp_idx[i], temp_idx[j] = temp_idx[j], temp_idx[i]

                vect1 = np.array(temp_centlst[1]) - np.array(temp_centlst[0])
                vect2 = np.array(temp_centlst[2]) - np.array(temp_centlst[0])

                vect1 = vect1 / np.linalg.norm(vect1)
                vect2 = vect2 / np.linalg.norm(vect2)



                coordinfo.loc[idx, 'coordinate'] = coord
                coordinfo.loc[idx, 'coord_idx'] = idx
                coordinfo.loc[idx, 'close2coord_idx'] = [temp_idx[1], temp_idx[2]]
                coordinfo.loc[idx, 'close2coord_cos_abs'] = abs(np.dot(vect1, vect2))
                coordinfo.loc[idx, 'average_vect'] = [vect1, vect2]
                coordinfo.loc[idx, 'radius'] = radlst[idx]

            # print('line185', coordinfo)



            while True:
                cur_screen_idx = copy.deepcopy(cur_screen_idx2)
                # print('line214',cur_screen_idx)

                if len(cur_screen_idx) < 3:
                    break 

                cos_lst = [ coordinfo.loc[idx, 'close2coord_cos_abs'] for idx in cur_screen_idx ]
                # print('line216', np.where(np.array(cos_lst) == np.max(cos_lst)), 'id', np.array(cur_screen_idx)[np.where(np.array(cos_lst) == np.max(cos_lst))])
                left_idx_lst = []



                if np.max(cos_lst) > dot_threshold:
                    labeled_idx1.append([])
                    labeled_idx_last_run = copy.deepcopy(labeled_idx1)

                    cur_std_coord_idx = np.array(cur_screen_idx)[np.where(np.array(cos_lst) == np.max(cos_lst))][0]
                    close_p_vect1 = coordinfo.loc[cur_std_coord_idx, 'average_vect'][0]
                    close_p_vect2 = coordinfo.loc[cur_std_coord_idx, 'average_vect'][1]
                    # print('line229', np.dot(close_p_vect1, close_p_vect2))
                    if np.dot(close_p_vect1, close_p_vect2) > 0:
                        cur_std_vect = (close_p_vect1 + close_p_vect2) / 2
                    else:
                        cur_std_vect = (close_p_vect1 - close_p_vect2) / 2
                    
                    labeled_idex_inline = [i for j in labeled_idx1 for i in j ]

                    if  cur_std_coord_idx in labeled_idex_inline or \
                        coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][0]  in labeled_idex_inline or \
                        coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][1]  in labeled_idex_inline:

                        left_idx_lst.append(cur_std_coord_idx) ## append 3 points to the 
                        left_idx_lst.append(coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][0])
                        left_idx_lst.append(coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][1])
                    
                    else:

                        labeled_idx1[-1].append(cur_std_coord_idx) ## append 3 points to the 
                        labeled_idx1[-1].append(coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][0])
                        labeled_idx1[-1].append(coordinfo.loc[cur_std_coord_idx, 'close2coord_idx'][1])



                        while True:
                            
                            dot_hist = []
                            for coordidx in cur_screen_idx: ## for every coord in sceen
                                '''
                                if coords vect close to std vect and one of the closest coord in list, add idx to label idx list
                                '''
        
                                for index in labeled_idx1[-1]:
                                    if index != coordidx:
                                        vect_from_cur_cent = np.array( np.array(coordinfo.loc[coordidx, 'coordinate']) - coordinfo.loc[index, 'coordinate'])
                                        vect_from_cur_cent_unit = vect_from_cur_cent / np.linalg.norm(vect_from_cur_cent)
                                        dot_value = abs(np.dot(vect_from_cur_cent_unit, cur_std_vect))
                                        dot_hist.append(dot_value)
                                        if dot_value > dot_threshold:
                                            # print('line256',np.linalg.norm(vect_from_cur_cent),)
                                            # print(('line257', coordinfo.loc[index, 'radius'], coordinfo.loc[coordidx, 'radius']))
                                            if np.linalg.norm(vect_from_cur_cent) < 3 * np.max((coordinfo.loc[index, 'radius'],  coordinfo.loc[coordidx, 'radius'])) : 
                                            # if np.linalg.norm(vect_from_cur_cent) < 2 * max(coordinfo.loc[index, 'radius'], coordinfo.loc[coordidx, 'radius']) :
                                                # print('line257-2', [i for j in labeled_idx1 for i in j ])
                                                if coordidx not in [i for j in labeled_idx1 for i in j ]:
                                                    labeled_idx1[-1].append(coordidx)
                                                    # if np.dot(vect_from_cur_cent, cur_std_vect) > 0:
                                                    #     cur_std_vect += vect_from_cur_cent
                                                    # else: cur_std_vect -= vect_from_cur_cent


                            ## panduan labeled_idx1 increase 
                        
                            

                            # print('line256',labeled_idx1)
                            cur_length = len([i for j in labeled_idx1 for i in j ])
                            last_length = len([i for j in labeled_idx_last_run for i in j ]) 
                            if cur_length <= last_length :
                                break

                            labeled_idx_last_run = copy.deepcopy(labeled_idx1)

                        # start new cession




                        if check_:
                            coordlst = [centerlst[i] for i in labeled_idx1[-1] ]
                            fig = plt.figure(figsize=(18,12))
                            ax = plt.subplot(111)
                            ax.imshow(self.mrc[int(self.mrc.shape[0]/2)])
                            ax.axis('off')

                            for num_ in range(len(coordlst)):
                                circle1 = plt.Circle((coordlst[num_][2], coordlst[num_][1]), 0.7, color = 'r', linewidth=2, fill = False )
                                # plt.gcf().gca().add_artist(circle1) 
                                plt.gcf().gca().add_patch(circle1) 
                                # add_patch  

                            plt.show()
                            plt.close()


                    # print('line286', [i for j in labeled_idx1 for i in j ])
                    cur_screen_idx2 = [idx for idx in cur_screen_idx if idx not in [i for j in labeled_idx1 for i in j ] and idx not in left_idx_lst]
                    
                    # print('line283',cur_screen_idx2)


                else: break  



        if len(cur_screen_idx) > 2:



            ## cur_screen_idx for screen in k mean ##
            # labeled_idx1 = labeled_idx1   labeled index 
            max_k = len(cur_screen_idx) - 1 # ensure every center count 1~2

            ## cur_screen_idx = cur_screen_idx
            print('line338', cur_screen_idx)
            centlst_for_kmean = [ coordinfo.loc[idx, 'coordinate'] for idx in cur_screen_idx]

            final_k = 1



            last_dist2kcent = float('inf')
            for k in range(1, max_k+1):
                Estimator = K_Means(k = k)
                Estimator.fit(centlst_for_kmean)
                # cur_dist2kcent = Estimator.closedist2()
                cur_dist2kcent = Estimator.elbowdist()
                if cur_dist2kcent > 0.8 * last_dist2kcent:
                    final_k = k - 1
                    break 
                last_dist2kcent = cur_dist2kcent
            
            Estimator = K_Means(k = final_k)
            Estimator.fit(centlst_for_kmean)
            
            label_pred = np.array([Estimator.predict(data) for data in centlst_for_kmean ])

            if check_:
                kmean_cent = list(Estimator.centers_.values())
                coordlst = kmean_cent
                fig = plt.figure(figsize=(18,12))
                ax = plt.subplot(111)
                ax.imshow(self.mrc[int(self.mrc.shape[0]/2)])
                ax.axis('off')

                for num_ in range(len(coordlst)):
                    circle1 = plt.Circle((coordlst[num_][2], coordlst[num_][1]), 0.7, color = 'r', linewidth=2, fill = False )
                    # plt.gcf().gca().add_artist(circle1) 
                    plt.gcf().gca().add_patch(circle1) 
                    # add_patch  

                plt.show()
                plt.close()
        

            for i in range(final_k):
                labeled_idx1.append([])

            for idx, label in enumerate(label_pred):
                labeled_idx1[-1-label].append(cur_screen_idx[idx])



        elif len(cur_screen_idx) == 2:
            coord1 = centerlst[cur_screen_idx[0]]
            coord2 = centerlst[cur_screen_idx[1]]
            r1 = radlst[cur_screen_idx[0]]
            r2 = radlst[cur_screen_idx[1]]
            if np.linalg.norm(np.array(coord1) - np.array(coord2)) > 3 * np.max((r1, r2)):
                labeled_idx1.append([cur_screen_idx[0]])
                labeled_idx1.append([cur_screen_idx[1]])
            else:
                labeled_idx1.append(cur_screen_idx)

        elif len(cur_screen_idx) == 1:
            labeled_idx1.append(cur_screen_idx)
        
        elif len(cur_screen_idx) == 0:
            pass


        self.labeled_center_index = labeled_idx1



# def main():
#     mask, mrc = read_testfile()
    
#     mito_seg = blob_classification(mask, mrc, centerlist)




# if __name__ == '__main__':

#     main() 



















# for keans

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(k=2)
    k_means.fit(x)
    # print('line349',k_means.centers_)
    # print('line350', list(k_means.centers_.values()))

    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    # print(k_means.clf_)
    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    predict = [[2, 1], [6, 9]]
    # print([k_means.predict(data) for data in predict ])
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()

