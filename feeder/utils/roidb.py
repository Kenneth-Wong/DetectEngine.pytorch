
import PIL
import numpy as np
from dataset.utils.imdb import imdb as IMDB
from dataset.utils.factory import get_imdb


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
      are useful for training. This function precomputes the maximum
      overlap, taken over ground-truth boxes, between each ROI and
      each ground-truth box. The class with maximum overlap is also
      recorded.
      """
    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]

    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    ratio_large = 2
    ratio_small = 0.5

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / height

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0
        ratio_list.append(ratio)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list) # from small to large
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1
    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True, **args):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        train_feeder_args = args['train_feeder_args']

        """Returns a roidb (Region of Interest database) for use in training."""
        if training and train_feeder_args['use_flipped']:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing data...')

        prepare_roidb(imdb)
        print('done')

        return imdb.roidb

    def get_roidb(imdb_name):
        train_args = args['train_args']
        imdb = get_imdb(imdb_name, **args)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(train_args['proposal_method'])
        print('Set proposal method: {:s}'.format(train_args['proposal_method']))
        roidb = get_training_roidb(imdb)

        # debug: mini dataset?
        if args['train_feeder_args']['debug']:
            roidb = roidb[:22]
        return roidb, imdb.classes

    roidbs = []
    for s in imdb_names.split('+'):
        roidb, classes = get_roidb(s)
        roidbs.append(roidb)
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = IMDB(imdb_names, classes, **args)
    else:
        imdb = get_imdb(imdb_names, **args)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index