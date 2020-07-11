import tensorflow as tf
from tensorflow.python.ops import array_ops

# focal loss with multi label
def focal_loss(target_tensor, prediction_tensor,classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes

    '''
    prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
    target_tensor is the label tensor, same shape as predcition_tensor
    '''
    import tensorflow as tf
    from tensorflow.python.ops import array_ops
    from keras import backend as K

    #1# get focal loss with no balanced weight which presented in paper function (4)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
    one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
    FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

    #2# get balanced weight alpha
    classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    total_num = float(sum(classes_num))
    classes_w_t1 = [ total_num / ff for ff in classes_num ]
    sum_ = sum(classes_w_t1)
    classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
    classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
    classes_weight += classes_w_tensor

    alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

    #3# get balanced focal loss
    balanced_fl = alpha * FT
    balanced_fl = tf.reduce_mean(balanced_fl)

    #4# add other op to prevent overfit
    # reference : https://spaces.ac.cn/archives/4493
    nb_classes = len(classes_num)
    fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes,
                                                                         prediction_tensor)
    # gp_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(prediction_tensor)/nb_classes, logits=prediction_tensor)
    # gp_loss=tf.reduce_mean(tf.reduce_sum(gp_loss, axis=1), name='loss')
    # fianal_loss = (1-e) * balanced_fl + e * gp_loss

    return fianal_loss
# def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
#     r"""Compute focal loss for predictions.
#         Multi-labels Focal loss formula:
#             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#     Args:
#      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing the predicted logits for each class
#      target_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing one-hot encoded classification targets
#      weights: A float tensor of shape [batch_size, num_anchors]
#      alpha: A scalar tensor for focal loss alpha hyper-parameter
#      gamma: A scalar tensor for focal loss gamma hyper-parameter
#     Returns:
#         loss: A (scalar) tensor representing the value of the loss function
#     """
#     sigmoid_p = tf.nn.sigmoid(prediction_tensor)
#     zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
#
#     # For poitive prediction, only need consider front part loss, back part is 0;
#     # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
#     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
#
#     # For negative prediction, only need consider back part loss, front part is 0;
#     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
#     return tf.reduce_sum(per_entry_cross_ent)