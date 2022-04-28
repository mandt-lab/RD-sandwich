These are results on high-resolution image compression from Section 6.4 of the paper.

`raw/` contains the raw .npz files from running the evaluation commands on the RD-UB VAEs, one for each combination of dataset (kodak, tecnick), model architecture ('resnet_vae', 'ms2020_vae'), and lambda value (0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001). Each .npz file contains the (bpp, mse, psnr, msssim) of individual decoded images (24 images in kodak, 100 images in tecnick).

`aggregate/` contains the aggregate (bpp, mse, psnr) for each dataset/model architecture, by averaging the results over all the images in a dataset for a fixed lambda.

