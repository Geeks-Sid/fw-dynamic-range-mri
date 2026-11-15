# caluclation from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4264049/

# img_path = 't1_to_t1ce.nii.gz'
# headmask_path = 'Head_Mask_revised.nii.gz'

import nibabel as nib
import numpy as np
import sys

import logging
log = logging.getLogger(__name__)

def get_a(sigma, Mn):
    x = Mn/sigma
    if x >= 3:
        a = 1
    else:
        a = 2.2 - (0.4*x)
    return a

def calc_dr(img_path, headmask_path):
    # ========================= initialize inputs =========================
    log.info('Loading input images')
    ## load image
    img = nib.load(img_path)
    img_data = img.get_fdata()
    # img_data = np.float32(img_data)
    img_data = np.float64(img_data)

    ## load head mask
    mask = nib.load(headmask_path)
    mask_data = mask.get_fdata()
    mask_data_reshaped = mask_data
    # mask_data_reshaped = np.transpose(mask_data, (1,2,0)) # not 2,1,0
    # mask_data_reshaped = np.flipud(mask_data_reshaped)
    masked_img_data = img_data*mask_data_reshaped
    ## get the image background
    background_bool = np.logical_not(mask_data_reshaped) # create a background mask (opposite of head mask)
    # new_img = nib.Nifti1Image(background_bool, img.affine, img.header)
    # nib.save(new_img, 'background_mask.nii.gz')
    img_data_background = img_data*background_bool
    # new_img = nib.Nifti1Image(img_data_background, img.affine, img.header)
    # nib.save(new_img, 'image_background.nii.gz')

    ## head-mask the input image
    # img_in_masked = img_data*mask_data_reshaped
    # new_img = nib.Nifti1Image(img_in_masked, img.affine, img.header)
    # nib.save(new_img, 'headmasked_input.nii.gz')

    # ========================= calculate the DR =========================
    ## set up static vars
    sigma = np.std(img_data_background) # noise standard deviation
    sigma_squared = sigma**2
    N = img_data.size # total number of image points
    # print(f'sigma = {sigma} / sigma^2 = {sigma_squared}')
    # print(f'Total number of image points: {N}')
    log.info('Calculating the dynamic range...')
    out = []
    for n in np.nditer(img_data): # for each image point, where n = magnitude signaln (loops over every voxel in image)
    # for n in np.nditer(masked_img_data): # for each image point, where n = magnitude signaln (loops over every voxel within the head mask)
        n_squared = n**2
        a = get_a(sigma, n)
        var1 = np.sign(n_squared - (a*sigma_squared))
        var2 = np.sqrt(abs(n_squared - (a*sigma_squared)))
        out.append(var1 * var2)

    DR = (1 / (sigma*np.sqrt(N))) * sum(out)
    # print(DR)
    return DR

if __name__ == "__main__":
    img_path = sys.argv[1]
    headmask_path = sys.argv[2]

    DR = calc_dr(img_path, headmask_path)
    log.info(f"Estimated dynamic range of the input image: {str(DR)}")
    print(f"Estimated dynamic range of the input image: {str(DR)}")
