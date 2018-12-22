import torch
import numpy as np
import numpy.random as npr
from scipy.misc import imread
from feeder.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
import cv2


def get_minibatch(roidb, num_classes, **args):
    """Given a roidb, construct a minibatch sampled from it."""
    train_args = args['train_args']
    train_feeder_args = args['train_feeder_args']
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(train_feeder_args['scales']),
                                    size=num_images)
    # assert (train_args['batch_size'] % num_images == 0), \
    #    'num_images ({}) must divide BATCH_SIZE ({})'. \
    #        format(num_images, train_args['batch_size'])

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, **args)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if train_feeder_args['use_all_gt']:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']

    return blobs


def _get_image_blob(roidb, scale_inds, **args):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    train_feeder_args = args['train_feeder_args']
    mix_args = args['mix_args']
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])

        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = train_feeder_args['scales'][scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, np.array(mix_args['pixel_means']), target_size,
                                        train_feeder_args['max_size'])
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


#####################################################################################################################

# for yolo
def get_size_minibatch(roidb, input_size, size_index, max_num_box=0, is_training=True):
    w_in, h_in = input_size[size_index]

    im = cv2.imread(roidb[0]['image'])
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.concatenate((im, im, im), axis=2)
    h_ori, w_ori = im.shape[:2]
    if is_training:
        boxes = roidb[0]['boxes'].copy()
        gt_classes = roidb[0]['gt_classes'].copy()
        boxes = np.hstack((boxes, gt_classes.reshape((boxes.shape[0], 1))))
        boxes[:, -1] -= 1
        all_boxes = np.zeros((max_num_box, boxes.shape[1]), dtype=np.float32)
        not_keep = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
        keep = np.where(not_keep == 0)[0]
        if len(keep) > 0:
            boxes = boxes[keep]
            num_box = min(boxes.shape[0], max_num_box)
            all_boxes[:num_box, :] = boxes[:num_box, :]
        else:
            num_box = 0

        im, trans_param = imcv2_affine_trans(im)
        scale, offs, flip = trans_param
        all_boxes[:num_box, :4] = offset_boxes(all_boxes[:num_box, :4], im.shape, scale, offs, flip)
        all_boxes[:num_box, :4][:, 0::2] *= (float(w_in) / float(im.shape[1]))
        all_boxes[:num_box, :4][:, 1::2] *= (float(h_in) / float(im.shape[0]))
        im = cv2.resize(im, (w_in, h_in))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = imcv2_recolor(im)
    else:
        im = cv2.resize(im, (w_in, h_in))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im / 255.
        num_box = 0
        all_boxes = np.array([1., 1., 1., 1., 1.], dtype=np.float32)

    return im, np.array([h_ori, w_ori, 1.]), all_boxes, num_box


def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]


def imcv2_recolor(im, a=.1):
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im


def offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x

    return boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
