This script is to segment insulin vesicles especially the vesicles which are attached with each other.
Input data include the SXT image(.mrc) and corresponding insulin vesicles masks(.tiff).
Noise reduction on SXT image with Gaussian filters.
Gather of insulin vesicle center coordinate candidates by finding local max intensity on the filtered SXT image.
Rank of candidates with intensity value.
Fit of spheres with different radius to these vesicle center. Since the space is represented in voxel, the fitting sphere also is a cluster of voxels.
Calculate the overlap of manual segmentation mask with the fitting sphere.
Set of label to voxels on mask by comparison of ratio of distance with radius to each voxel center. the minimum ratio defines the classification of current voxel.

