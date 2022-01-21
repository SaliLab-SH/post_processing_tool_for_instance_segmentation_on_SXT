from enum import EnumMeta
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


dirpath = os.path.dirname(__file__)
os.chdir(dirpath)
sys.path.append(os.pardir)
print('curpath', os.getcwd())


def obtain_seq():
    random.seed(7)  ## even no seed, label is the same

    mrc_img = mrcfile.open(f'E:/PBC_data/datasets/for_24_datasets/Stevens_pancreatic_INS_1E_25-10_30min_822_4_pre_rec.mrc', permissive=True).data
    img_original = tifffile.imread(f'E:/PBC_data/datasets/tiff_24datasets_organelles/822_4_isg_label.tiff')
    img_instance = tifffile.imread(f'F:/salilab_localdata/vesicle_instance_mask/822_4_instance_mask.tiff')
    img_label, _ =ndi.label(img_original)
    # print()
    difference = img_label - img_instance

    print(np.max(img_original), np.max(img_instance), np.max(img_label),np.max(difference))
    # print('line23', list(set(list(difference.reshape(1,-1))[0])))
    single_isg_lst = []
    label_idxlst = [i for i in range(1, np.max(img_label)+1)]
    # for idx in range(1, np.max(img_label) + 1):
    #     # print('line24', difference[np.where(img_label == idx)])
    #     index_lst = list(set(difference[np.where(img_label == idx)]))    
    #     print('line25',idx, index_lst)
    #     if len(index_lst) == 1:
    #         single_isg_lst.append(idx)


    def get_single_instance(idx):
        index_lst = list(set(difference[np.where(img_label == idx)])) 
        if len(index_lst) == 1:
            print('line39',idx)
            return idx


    aa = list(map(get_single_instance, label_idxlst ))
    aa = [i for i in aa if i]
    aa = sorted(aa)
    print(aa)



##  [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 189, 191, 192, 193, 194, 195, 197, 198, 200, 202, 203, 204, 205, 206, 207, 209, 211, 212, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 309, 311, 313, 314, 315, 316, 317, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 332, 333, 334, 335, 337, 338, 339, 340, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 382, 383, 384, 385, 388, 389, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 405, 406, 407, 408, 409, 410, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 429, 431, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 463, 464, 465, 467, 468, 469, 470, 472, 473, 474, 476, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 496, 498, 499, 500, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 514, 515, 517, 519, 520, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 539, 540, 541, 542, 543, 544, 545, 546, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563, 564, 565, 567, 569, 570, 571, 572, 573, 575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 587, 588, 589, 591, 593, 595, 596, 597, 598, 599, 600, 602, 603, 604, 605, 607, 608, 609, 610, 611, 612, 613, 614, 615, 618, 619, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 634, 636, 637, 638, 639, 640, 641, 642, 643, 645, 646, 648, 649, 650, 652, 653, 654, 656, 657, 658, 660, 661, 662, 663, 664, 665, 667, 669, 670, 671, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 686, 687, 689, 690, 691, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724]


def generate_mask(name_idx):
    # np.random.seed(8)
    img_original = tifffile.imread(f'E:/PBC_data/datasets/tiff_24datasets_organelles/822_4_isg_label.tiff')
    mrc_img = mrcfile.open(f'E:/PBC_data/datasets/for_24_datasets/Stevens_pancreatic_INS_1E_25-10_30min_822_4_pre_rec.mrc', permissive=True).data
    mrc_img = mrc_img * 27.161
    img_label, _ =ndi.label(img_original)
    single_isntance_idxlst =[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 189, 191, 192, 193, 194, 195, 197, 198, 200, 202, 203, 204, 205, 206, 207, 209, 211, 212, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 309, 311, 313, 314, 315, 316, 317, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 332, 333, 334, 335, 337, 338, 339, 340, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 382, 383, 384, 385, 388, 389, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 405, 406, 407, 408, 409, 410, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 429, 431, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 463, 464, 465, 467, 468, 469, 470, 472, 473, 474, 476, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 496, 498, 499, 500, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 514, 515, 517, 519, 520, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 539, 540, 541, 542, 543, 544, 545, 546, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563, 564, 565, 567, 569, 570, 571, 572, 573, 575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 587, 588, 589, 591, 593, 595, 596, 597, 598, 599, 600, 602, 603, 604, 605, 607, 608, 609, 610, 611, 612, 613, 614, 615, 618, 619, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 634, 636, 637, 638, 639, 640, 641, 642, 643, 645, 646, 648, 649, 650, 652, 653, 654, 656, 657, 658, 660, 661, 662, 663, 664, 665, 667, 669, 670, 671, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 686, 687, 689, 690, 691, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724]



    idxlst = np.random.choice(single_isntance_idxlst, 5, replace=False)
    # dixlst = [170, 464, 52, 79, 640]
    print(idxlst)  # [170, 464, 52, 79, 640]
    generated_tiff = np.zeros([50,100,100])
    generated_mrc = np.zeros_like(generated_tiff)
    generated_mrc = np.random.normal(0.25,0.02, (generated_tiff.shape[0], generated_tiff.shape[1], generated_tiff.shape[2]))
    # generated_mrc[np.where(generated_mrc == 0)] = random.gauss(0.25, 2.62e-3)

    # plt.imshow(generated_mrc[25])
    # plt.show()
    # plt.close()

    shape_ = generated_tiff.shape

    for ii, index in enumerate(idxlst):
        coords = np.array(np.where(img_label == index))
        x_min, x_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        z_min, z_max = np.min(coords[2]), np.max(coords[2])

        probable_lst = []
        for x in range(0+1, shape_[0] - (x_max - x_min + 1) -1):
            for y in range(0+1, shape_[1] - (y_max - y_min +1) -1):
                for z in range(0+1, shape_[2] - (z_max - z_min +1) -1):
                    probable_lst.append([x,y,z])

        print('line85',index, len(probable_lst))

        n = 0
        pick_lst = random.sample(probable_lst, len(probable_lst))

        for i, num in enumerate(pick_lst):
            
            # x_shift = np.random.randint(0+1, shape_[0] - (x_max - x_min + 1) -1)
            # y_shift = np.random.randint(0+1, shape_[1] - (y_max - y_min +1) -1)
            # z_shift = np.random.randint(0+1, shape_[2] - (z_max - z_min +1) -1)
            # coords_x = coords[0] - x_min + x_shift
            # coords_y = coords[1] - y_min + y_shift
            # coords_z = coords[2] - z_min + z_shift
            coords_x = coords[0] - x_min + pick_lst[i][0]
            coords_y = coords[1] - y_min + pick_lst[i][1]
            coords_z = coords[2] - z_min + pick_lst[i][2]

            coord_new = [coords_x, coords_y, coords_z]

            temp_tiff = np.zeros_like(generated_tiff)
            check_tiff = copy.deepcopy(generated_tiff)

            n += 1

            if np.max(generated_tiff[tuple(coord_new)]) == 0:  # no overlap
                temp_tiff[tuple(coord_new)] = index
                check_tiff = check_tiff + temp_tiff

                temp_label_mask, _ = ndi.label(check_tiff)
                if np.max(temp_label_mask) == 1:
                    generated_tiff = generated_tiff + temp_tiff
                    generated_mrc[tuple(coord_new)] = mrc_img[tuple([coords[0], coords[1], coords[2]])]
                    break
            
            # if n > 9999:
            #     print('out of time ')
            #     raise RuntimeError('too long loop')
    
        else:
            print('failed to add ', index)

    idex_lst = list(set(list(generated_tiff.reshape(1,-1))[0]))
    idex_lst.remove(0)


    print('line118', idex_lst)
    for i, idx in enumerate(idex_lst):
        idx = int(idx)
        generated_tiff[np.where(generated_tiff == idx)] = i+1
        

    tifffile.imsave(f'./benchmark/vesicle_sample_5isg_{name_idx}.tiff', generated_tiff)
    tifffile.imsave(f'./benchmark/vesicle_sample_5isg_{name_idx}_mrc.tiff', generated_mrc)
    

    print('done')





def main():
    # obtain_seq()
    idxlist = [i for i in range(10)]
    # idxlist = [1]
    def func(idx):
        generate_mask(idx)

    list(map(func, idxlist))


if __name__ == '__main__':
    main()