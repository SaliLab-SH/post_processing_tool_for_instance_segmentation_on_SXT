## 
from ctypes import cdll
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import sys, os
from numpy.core.defchararray import center
from numpy.core.fromnumeric import argmax, argmin
from numpy.core.numeric import True_
from numpy.core.shape_base import _accumulate, vstack
from numpy.ma.core import concatenate 
import tifffile
from scipy import ndimage as ndi
import skimage
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter,gaussian_laplace
from multiprocessing import Pool
from multiprocessing import Process, freeze_support
from scipy.spatial.distance import cdist
import copy
from numba import jit

dirpath = os.path.dirname(__file__)
os.chdir(dirpath)
sys.path.append(os.pardir)
print('curpath', os.getcwd)




def generation_mrc():
    ## generate a mask called file to app

    #----------build test dataset file-----------
    test_datafile_name = f'./test_datasets'
    if not os.path.exists(test_datafile_name):
        os.makedirs(test_datafile_name)




    def read_image(datasetnum, check_ = 0):
    
    #     mainpath = r'./data'
        mainpath = r'E:/PBC_data/datasets/tiff_24datasets_organelles'
        for maindir, subdir, file_name_list in os.walk(mainpath, topdown=False):
            filelist = np.array(file_name_list)

        for name in filelist:
            if datasetnum in name and 'isg' in name and 'tiff' in name:
                isg_tif_name = f'{mainpath}/{name}'
                print('isg_tif_name',isg_tif_name)
            elif datasetnum in name and 'mito' in name and 'tiff' in name: 
                mito_tif_name = f'{mainpath}/{name}'
                print('mito_tif_name ',mito_tif_name )
            elif datasetnum in name and  'wholecell' in name and 'tiff' in name:
                wholecell_tif_name = f'{mainpath}/{name}'
                print('wholecell_tif_name', wholecell_tif_name)
            else:
                pass
        
        mrcpath = r'E:/PBC_data/datasets/for_24_datasets' 
        for maindir, subdir, file_name_list in os.walk(mrcpath, topdown=False):
            filelist = np.array(file_name_list)
        for name in filelist:
            if datasetnum in name and  '.mrc' in name:
                mrc_name = f'{mrcpath}/{name}'
                print(mrc_name)
            else:
                pass
            
            
            
        tiff_isg = tifffile.imread(isg_tif_name)
        tiff_mito = tifffile.imread(mito_tif_name)
        tiff_wholecell = tifffile.imread(wholecell_tif_name) 
        mrc = mrcfile.open(mrc_name, permissive=True).data

        mrc = mrc * 27.161

        # if check_:
        #     fig, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(20,20))
        #     ax1.imshow(tiff_isg[232, :, :] )
        #     ax2.imshow(tiff_mito[232, :, :])
        #     ax3.imshow(tiff_wholecell[232, :, :])
        #     ax4.imshow(mrc[232, :, :])
        #     plt.show()
        #     plt.close()
        
    #     return datasetnum
        return tiff_isg, tiff_mito, tiff_wholecell, mrc


        ## check data here 
    datasetlsts = ['822_4',]
    # datasetlsts = ['766_2','766_5','766_7','766_8','766_10',]
    # datasetlsts = ['766_2','766_5','766_7','766_8','766_10','766_11','769_5','769_7','783_5','783_11', '783_12','784_4','784_5','784_6','784_7','785_7','822_4','822_6','822_7','842_12','842_13', '842_17','931_9','931_14']

    datasetlsts2 = list(map(read_image, datasetlsts))

    tiff_isg = datasetlsts2[0][0]
    tiff_mito = datasetlsts2[0][1]
    tiff_wholecell = datasetlsts2[0][2]
    raw_mrc = datasetlsts2[0][3]

    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(20,20))
    ax1.imshow(tiff_isg[232, :, :] )
    ax2.imshow(tiff_mito[232, :, :])
    ax3.imshow(tiff_wholecell[232, :, :])
    ax4.imshow(raw_mrc[232, :, :])
    plt.show()
    plt.close()


    ## extract granules as sample 
    instance_tiff_isg = skimage.morphology.label(tiff_isg)

    plt.imshow(instance_tiff_isg[189])
    plt.show()

    tiff_maxindex_ =np.max(instance_tiff_isg)
    print(tiff_maxindex_)

    

    final_tiff = np.zeros_like(tiff_isg)

    # blobs_tiff = list(map(granule_separation_main,range(1,5)))    #range(1,tiff_maxindex_ + 1)))



    min_volume = 10
    min_rad = 1.5
    # n_idealball, v_idealball, uniqueR = create_ideal_sphere(R = 20)


    for granule_index in range(132, 133): 
        print('current granule index', granule_index)
        mask_index = granule_index
        temp_tiff = np.zeros_like(tiff_isg)
        temp_tiff[np.where(instance_tiff_isg == granule_index)] = 1
        coords = np.where(instance_tiff_isg == granule_index)
        x_ = int(np.mean(coords[0]))
        y_ = int(np.mean(coords[1]))
        z_ = int(np.mean(coords[2]))
        #     plt.imshow(temp_tiff[x_])
        volume = np.sum(temp_tiff)
        
        roi01 = ( slice(max(0,int(np.min(coords[0]))), min(instance_tiff_isg.shape[0], 1+int(np.max(coords[0])) )),slice(max(0,int(np.min(coords[1]))-10), min(instance_tiff_isg.shape[1], int(np.max(coords[1])) +10)), slice(max(0,int(np.min(coords[2]))-10), min(instance_tiff_isg.shape[2], int(np.max(coords[2]))+10 ))) 
        anchorcoord = [max(0,int(np.min(coords[0]))),max(0,int(np.min(coords[1]))-10), max(0,int(np.min(coords[2]))-10)] ## for reversing blobs together
    #     print('anchorcoord',anchorcoord)
    #     print('roi',roi01)
        
        temp_isg_partial_tiff = temp_tiff[roi01]
        temp_isg_partial_mrc = raw_mrc[roi01]
    
        shape_ = temp_isg_partial_tiff.shape
        # for i in range(int(shape_[0])):
        # for i in range(1):
        #     plt.imshow(temp_isg_partial_tiff[8])  
        #     plt.show()
        #     plt.close()
        #     plt.imshow(temp_isg_partial_mrc[8])  
        #     plt.show()
        #     plt.close()

        test_mask = f'{test_datafile_name}/822_4_test_mask.tiff'
        test_mrc = f'{test_datafile_name}/822_4_test_mrc.tiff'

        if not os.path.exists(test_mask):  
            tifffile.imsave(test_mask, temp_isg_partial_tiff)
            tifffile.imsave(test_mrc, temp_isg_partial_mrc)
            print('saved.')
        else:
            print('file already exist')

    pass


def read_testfile(check_ = 0):
    datapath = f'E:/PBC_data/scripts/granule-separation/test_datasets'
    mask = tifffile.imread(f'{datapath}/822_4_test_mask.tiff')

    if check_:
        plt.imshow(mask[8])
        plt.show()
        plt.close()

    mrc = tifffile.imread(f'{datapath}/822_4_test_mrc.tiff')
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


def seperate_mask(label_mask, mrc, seqlst):

    # print('line217', np.max(label_mask))
    labellst = seqlst
    
    def return_maskrange(label):        
        coords = np.where(label_mask == label)
        shape_ = label_mask.shape
        # print(shape_)

        x_min, x_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        z_min, z_max = np.min(coords[2]), np.max(coords[2])

        range = (slice(np.max([x_min-5, 0]), np.min([x_max+1+5, shape_[0]])), 
                 slice(np.max([y_min-5, 0]), np.min([y_max+1+5, shape_[1]])), 
                 slice(np.max([z_min-5, 0]), np.min([z_max+1+5, shape_[2]])))
        return range

    rangelst = list(map(return_maskrange, labellst))

    # print('line232', rangelst)
    def return_tinymask(idx):
        label = labellst[idx]
        coords = np.where(label_mask == label)
        temp_mask = np.zeros_like(label_mask)
        temp_mask[coords] = label
        return [temp_mask[rangelst[idx]], mrc[rangelst[idx]]]
        
    
    masklst = list(map(return_tinymask, [i for i in range(len(labellst))]))

    return masklst, rangelst



def discrete_sphere(max_r, min_r):
    cube =  np.ones([int(max_r * 2 + 1 + 2), int(max_r * 2 + 1 + 2), int(max_r * 2 + 1 + 2)])
    # ball_ = skimage.morphology.ball(max_r)
    coords_ball = np.where(cube == 1)
    x_ball = int(np.mean(coords_ball[0]))
    y_ball = int(np.mean(coords_ball[1]))
    z_ball = int(np.mean(coords_ball[2]))
    cent_coord = np.array([[x_ball, y_ball, z_ball]])
    # print(cent_coord.shape)
    coords_allball = np.array(coords_ball).T
    # print(coords_allball.shape)
    dist_lst = np.array(cdist(cent_coord, coords_allball, metric='euclidean')[0])

    r_lst = np.array(sorted(list(set(dist_lst)),reverse=False))
    r_lst = r_lst[ np.where(r_lst > min_r)]
    r_volume_lst = [len(np.where(dist_lst <= r)[0]) for r in r_lst]



    # print(r_volume_lst)
    return np.concatenate((np.array([r_lst]), np.array([r_volume_lst])), axis = 0) 


def get_rad(centcoord, mask, r_seq, theta):
    r_lst = r_seq[0]
    r_volume = r_seq[1]

    all_coords = np.array(np.where(mask != 0)).T
    dist = cdist(np.array([centcoord]), all_coords)[0]
    
    mask_volumes = [ len(np.where(dist <= d )[0]) for d in r_lst ]
    ratiolst = [ mask_volumes[i]/r_volume[i] for i in range(len(r_lst))]

    # increased_ratiolst = [ ((mask_volumes[i+1] - mask_volumes[i])/(r_volume[i+1] - r_volume[i])) * (-2 * r_lst[i]**(-3)) for i in range(len(r_lst)-1)]
    # volume_lst = [ ((r_volume[i+1] - r_volume[i])) * (-2 * r_lst[i]**(-3)) for i in range(len(r_lst)-1)]
    # volume_substract_lst = [ ((mask_volumes[i+1] - mask_volumes[i]) - (r_volume[i+1] - r_volume[i])) for i in range(len(r_lst)-1)]
    # ratiolst_rr = [ (mask_volumes[i]/r_volume[i]) * ( r_lst[i]**(-2)) for i in range(len(r_lst)-1)]
    # ratiolst_r = [ (mask_volumes[i]/r_volume[i]) * ( r_lst[i]**(-1)) for i in range(len(r_lst)-1)]

    if np.max(ratiolst) < theta:
        return 0
    else:
        idx = np.where(np.array(ratiolst) >= theta)[0].shape[0]
        r = r_lst[idx]
        return r



def blob_fit(mask,  mrc, min_r = 1.5, theta = 0.8, check_ = False):  
    mask[np.where(mask !=0)] = 1
    n = int(mrc.shape[0]/2) #5
    sigmalst = list(np.arange(1, 10, 1) ) ##steps here
    # print(sigmalst)

    blobs_lst = []
    for sigma in sigmalst:
        filtered_mrc  = gaussian_filter(mrc, sigma)
        temp_blobs_lst = peak_local_max(filtered_mrc, footprint=np.ones((3,) * (mask.ndim)), exclude_border=True)
        blobs_lst.extend(temp_blobs_lst)

        # print('line304',len(blobs_lst))

    if check_:
        fig = plt.figure(figsize=(18,12))
        ax = plt.subplot(111)
        ax.imshow(mrc[n])
        ax.axis('off')

        for num_ in range(len(blobs_lst)):
            circle1 = plt.Circle((blobs_lst[num_][2], blobs_lst[num_][1]), 0.7, color = 'r', linewidth=2, fill = False )
            # plt.gcf().gca().add_artist(circle1) 
            plt.gcf().gca().add_patch(circle1) 
            # add_patch  


    if not blobs_lst:
        # print('line326')
        if np.sum(mask) > 15:
            

            if check_:
                print('line326-1')
                fig = plt.figure(figsize=(18,12))
                ax = plt.subplot(111)
                ax.imshow(mask[n])
        # ax.axis('off')
            return mask # np.zeros_like(mask)
        else: 
            
            if check_:
                print('line326-2')
                fig = plt.figure(figsize=(18,12))
                ax = plt.subplot(111)
                ax.imshow(np.zeros_like(mask)[n])
            return np.zeros_like(mask)

    # print(blobs_lst)
    # print('line300',len(blobs_lst))

    coord_intensity_lst = []
    coord_edt_lst = []
    edtmap = ndi.distance_transform_edt(ndi.binary_dilation(mask))
    
    # plt.imshow(edtmap[n])
    # plt.show()
    # plt.close()

    for coord in blobs_lst:
        coord_intensity_lst.append(mrc[coord[0]][coord[1]][coord[2]])
        coord_edt_lst.append(edtmap[coord[0]][coord[1]][coord[2]])



    ## sort coords in priority of LAC value
    for i in range(len(coord_intensity_lst)-1):
        for j in range(i, len(coord_intensity_lst)):
            if coord_intensity_lst[i] < coord_intensity_lst[j]:
                coord_intensity_lst[i], coord_intensity_lst[j] = coord_intensity_lst[j], coord_intensity_lst[i]
                coord_edt_lst[i], coord_edt_lst[j] = coord_edt_lst[j], coord_edt_lst[i]
                blobs_lst[i], blobs_lst[j] = blobs_lst[j], blobs_lst[i]


    max_r = np.max(coord_edt_lst)//1 + 2 + 1
    r_seq = discrete_sphere(max_r, min_r)


    
    ## match rad for coord
    final_cent_coord_lst = []
    final_cent_rad_lst = []
    iterated_mask = copy.deepcopy(mask)
    for idx, coord in enumerate(blobs_lst):
        # print('line335', idx, coord)
        if coord_edt_lst[idx] == 0:
            # print(111)
            continue

        if len(final_cent_coord_lst) != 0:
            dist = cdist(np.array([coord]), np.array(final_cent_coord_lst))[0]
            diff = dist - np.array(final_cent_rad_lst)
            test_temp = [i < 0 for i in diff]
            # print(222)
            if True in test_temp:
                continue

        ##  add theta adjust the range 
        r = get_rad(coord, mask, r_seq ,theta)


        if r > 0:
            if check_:
                print('line347--', idx, coord)  #get coord position

            final_cent_coord_lst.append(coord)
            final_cent_rad_lst.append(r) 

            temp_edt = np.ones_like(iterated_mask)
            temp_edt[coord[0]][coord[1]][coord[2]] = 0
            temp_edt = ndi.distance_transform_edt(temp_edt)
            iterated_mask[np.where(temp_edt < r)] = 0
            # plt.imshow(iterated_mask[n])
            # plt.show()
            # plt.close()
    del iterated_mask

    if check_:
        fig = plt.figure(figsize=(18,12))
        ax = plt.subplot(111)
        ax.imshow(mask[n])
        # ax.axis('off')
        

        for num_ in range(len(final_cent_coord_lst)):
            circle1 = plt.Circle((final_cent_coord_lst[num_][2], final_cent_coord_lst[num_][1]), final_cent_rad_lst[num_] , color = 'r', linewidth=2, fill = False )
            # plt.gcf().gca().add_artist(circle1) 
            plt.gcf().gca().add_patch(circle1) 
            # add_patch                   
        plt.show()
        plt.close()
            

    if not final_cent_coord_lst:
        # print('line415')
        if np.sum(mask) > 15:
            
            if check_:
                print('line415-1')
                fig = plt.figure(figsize=(18,12))
                ax = plt.subplot(111)
                ax.imshow(mask[n])
            return mask # np.zeros_like(mask)
        else: 
            
            if check_:
                print('line415-2')
                fig = plt.figure(figsize=(18,12))
                ax = plt.subplot(111)
                ax.imshow(np.zeros_like(mask)[n])
            return np.zeros_like(mask)

    
    ### separate mask
    instance_mask = np.zeros_like(mask)

    coordlst = np.array(np.where(mask > 0)).T
    idx_lst = [i for i in range(len(coordlst))]

    def set_label(idx):
        coord = coordlst[idx]
        distt = cdist( np.array([coord]), np.array(final_cent_coord_lst))
        ratio_d = distt/ np.array(final_cent_rad_lst)

        label = argmin([ratio_d])
        return label+1

    label_lst = list(map(set_label, idx_lst))
    

    for idx in idx_lst:
        coord = coordlst[idx]
        instance_mask[coord[0]][coord[1]][coord[2]] = label_lst[idx]



    # for i in range(instance_mask.shape[0]):
    #     fig = plt.figure()
    #     ax = plt.subplot(121)
    #     ax.imshow(instance_mask[i])
    #     ax.set_title(f'mask i = {i}')

    #     ax = plt.subplot(122)
    #     ax.imshow(mrc[i])
    #     ax.set_title(f'mrc i = {i}')

    #     plt.show()
    #     plt.close()

    return instance_mask




def main():
    mask, mrc = read_testfile(1)

    # mask = tifffile.imread(f'E:/PBC_data/datasets/tiff_24datasets_organelles/822_4_isg_label.tiff')
    # mrc = mrcfile.open(r'E:/PBC_data/datasets/for_24_datasets/Stevens_pancreatic_INS_1E_25-10_30min_822_4_pre_rec.mrc' , permissive=True).data
    # plt.imshow(mask[232])
    # plt.show()
    # plt.close()
    # plt.imshow(mrc[232])
    # plt.show()
    # plt.close()

    batch_num = 1


    label_mask,_ = ndi.label(mask)

    index_seq = [i for i in range(1, np.max(label_mask)+1)]
    batch_idx = [ index_seq[batch_num * i : batch_num * (i+1) ] for i in range(len(index_seq)//batch_num+1)]
    if not batch_idx[-1]:
        batch_idx = batch_idx[:-1]

    print('line470', batch_idx)
    batch_accumulate_seq = 0

    instance_mask = np.zeros_like(mask)
    
    for index, seqlst in enumerate(batch_idx):

        masklst, rangelst = seperate_mask(label_mask, mrc,  seqlst)
        idx_lst = [i for i in range(len(masklst))]
        # print('line461', seqlst)

        def set_fitmask(idx):
            test_mask, test_mrc = masklst[idx][0], masklst[idx][1]
            instance_tinymask = blob_fit(test_mask, test_mrc, check_=False)

            return instance_tinymask

        instance_tinymask_lst = list(map(set_fitmask, idx_lst)) 

        label_seq = [ np.max(mask) for mask in instance_tinymask_lst ]

        accumulate_seq = [np.sum(label_seq[:i]) + batch_accumulate_seq for i in range(len(label_seq))]
        # print('line461.5 label seq ',label_seq)
        # print('line462 accumulate_seq',accumulate_seq)



        
        for idx in idx_lst:
            temp_mask = np.zeros_like(mask)
            accumu_label_mask = np.zeros_like(instance_tinymask_lst[idx])
            accumu_label_mask[np.where(instance_tinymask_lst[idx] != 0)] = accumulate_seq[idx]
            temp_mask[rangelst[idx]] = instance_tinymask_lst[idx] + accumu_label_mask
            instance_mask += temp_mask
            del temp_mask, accumu_label_mask
        
        batch_accumulate_seq += np.sum(label_seq)
        print('proceded num', seqlst[-1])
        print('line505', np.max(instance_mask))


    print('line472', np.max(instance_mask))
    # print('line500', 'label on instance mask', list(set(list(instance_mask.reshape(1,-1))[0])))
    tifffile.imsave('test_instance_mask.tiff', instance_mask)
    # plt.imshow(instance_mask[232])
    plt.imshow(instance_mask[int(instance_mask.shape[0]/2)])
    plt.show()
    plt.close()


    print('done for all')


if __name__ == '__main__':
    main()
    # generation_mrc()
    # mask, mrc = read_testfile()


