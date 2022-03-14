strftime_format = "%Y_%m_%d~%H_%M_%S"


# the --dataset cmdline arg accepts a glob key defined here:
dataset_to_globs = {
    'cocotrain': 'data/my_coco_train2017/*.png',
    'kodak': 'data/kodak/kodim*.png',
    'tecnick': 'data/Tecnick_TESTIMAGES/RGB/RGB_OR_1200x1200/*.png',
}


cmdline_arg_abbr = {
    # UB model
    'num_filters': 'F',
    'latent_channels': 'C',
    # LB model
    'batchsize': 'k',
    'num_Ck_samples': 'M',
}

# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
biggan_class_names_to_ids = {
    'basenji': 253,
    'beagle': 162,
    'jay': 17,
    'magpie': 18,
}
