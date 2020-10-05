import tensorflow as tf
import math
import numpy as np

from fasterflow import utils

#----------------------------------------------------------------------------
# GAN losses

## Wasserstein loss
def Wasserstein(real_outputs, fake_outputs):
    with tf.compat.v1.name_scope('WassersteinLoss'):
        gen_loss = -fake_outputs
        disc_loss = fake_outputs - real_outputs
        return gen_loss, disc_loss

## Relativistic losses, defined in https://arxiv.org/abs/1807.00734
def RelativisticAverageBCE(real_outputs, fake_outputs):
    with tf.compat.v1.name_scope('RelativisticBCELoss'):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        f1 = lambda x: cross_entropy(tf.compat.v1.ones_like(x), x)
        f2 = lambda x: cross_entropy(tf.compat.v1.zeros_like(x), x)
        rf = real_outputs - tf.compat.v1.math.reduce_mean(fake_outputs)
        fr = fake_outputs - tf.compat.v1.math.reduce_mean(real_outputs)
        gen_loss = tf.compat.v1.math.reduce_mean(f2(rf)+f1(fr))
        disc_loss = tf.compat.v1.math.reduce_mean(f1(rf)+f2(fr))
        return gen_loss, disc_loss 

def RelativisticAverageHinge(real_outputs, fake_outputs):
    with tf.compat.v1.name_scope('RelativisticHingeLoss'):
        f1 = lambda x: tf.compat.v1.nn.relu(1-x)
        f2 = lambda x: tf.compat.v1.nn.relu(1+x)
        rf = real_outputs - tf.compat.v1.math.reduce_mean(fake_outputs)
        fr = fake_outputs - tf.compat.v1.math.reduce_mean(real_outputs)
        gen_loss = tf.compat.v1.math.reduce_mean(f2(rf)+f1(fr))
        disc_loss = tf.compat.v1.math.reduce_mean(f1(rf)+f2(fr))
        return gen_loss, disc_loss

def GradientPenalty(discriminator, image_inputs, fake_images, minibatch_size, wgan_target=1.0, wgan_lambda=10.0):
    with tf.compat.v1.name_scope('GradientPenalty'):
        mixing_factors = tf.compat.v1.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
        mixed_images = utils.lerp(image_inputs, fake_images, mixing_factors)
        mixed_loss = tf.compat.v1.math.reduce_sum(discriminator(mixed_images))
        mixed_grads = tf.compat.v1.gradients(mixed_loss, [mixed_images])
        mixed_norms = tf.compat.v1.math.sqrt(tf.compat.v1.math.reduce_sum(tf.compat.v1.math.square(mixed_grads), axis=[1,2,3]))
        gradient_penalty = tf.compat.v1.math.square(mixed_norms-wgan_target)
        return gradient_penalty * (wgan_lambda / (wgan_target**2))

def GradientPenaltyMSG(discriminator, image_inputs, fake_images, minibatch_size, wgan_target=1.0, wgan_lambda=10.0):
    with tf.compat.v1.name_scope('GradientPenalty'):
        mixing_factors = tf.compat.v1.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
        mixed_images = [utils.lerp(image_inputs[i], fake_images[i], mixing_factors) for i in range(len(image_inputs))]
        mixed_loss = tf.compat.v1.math.reduce_sum(discriminator(mixed_images))
        mixed_grads = tf.compat.v1.gradients(mixed_loss, mixed_images)
        mixed_norms = tf.compat.v1.math.sqrt(tf.compat.v1.math.reduce_sum(tf.compat.v1.math.square(mixed_grads), axis=[1,2,3]))
        gradient_penalty = tf.compat.v1.math.square(mixed_norms-wgan_target)
        return gradient_penalty * (wgan_lambda / (wgan_target**2))

def EpsilonPenalty(real_outputs):
    with tf.compat.v1.name_scope('EpsilonPenalty'):
        epsilon_penalty = tf.compat.v1.math.square(real_outputs)
        return epsilon_penalty * 1e-3

#----------------------------------------------------------------------------
# Recognition Losses

#############################################################################
# Triplet losses
#############################################################################
def triplet_loss(embs, alpha=0.2):
    """
    This loss is dependant on the embedding order. The embeddings are to be
    a sequence of (anchor,positive,negative).
    """
    embs1 = embs[0::3]
    embs2 = embs[1::3]
    embs3 = embs[2::3]
    
    dist_same = tf.reduce_sum(tf.square(embs1-embs2), axis=-1)
    dist_diff = tf.reduce_sum(tf.square(embs1-embs3), axis=-1)
    return tf.reduce_mean(tf.nn.relu(dist_same-dist_diff+alpha))

def triplet_accuracy(embs, alpha=0.2):
    embs1 = embs[0::3]
    embs2 = embs[1::3]
    embs3 = embs[2::3]
    
    dist_same = tf.reduce_sum(tf.square(embs1-embs2), axis=-1)
    dist_diff = tf.reduce_sum(tf.square(embs1-embs3), axis=-1)
    return tf.reduce_mean(tf.cast(dist_same+alpha<dist_diff, tf.float32))

def general_triplet_loss(embs, labels, minibatch_size, alpha=0.2):
    """
    NOTE: In order for this loss to work properly, it is prefered that 
    labels contains several repetitions. In other word:
    len(np.unique(labels))!=len(labels)
    Hence, do not shuffle the dataset!
    """
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix, inner product of the classes
    class_mat = tf.matmul(classes, tf.transpose(classes))
    # Create mask for the same and different classes
    # The class matrix is symetric, so only the upper/lower triangular matrix
    # is important.
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = dist_same - dist_diff + alpha
    nbof_valid_triplets = tf.reduce_sum(tf.cast(tf.greater(loss,1e-16),tf.float32))
    return tf.reduce_sum(tf.nn.relu(loss))/(nbof_valid_triplets+1e-16)

def general_triplet_accuracy(embs, labels, minibatch_size, alpha=0.2):
    """
    NOTE: In order for this accurcacy to work properly, it is prefered that 
    labels contains several repetitions. In other word:
    len(np.unique(labels))!=len(labels)
    Hence, do not shuffle the dataset!
    """
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix
    class_mat = tf.matmul(classes, tf.transpose(classes))
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = dist_same - dist_diff + alpha
    return tf.reduce_mean(tf.cast(loss<0, tf.float32))

def adaptive_triplet_loss(embs, labels, minibatch_size, alpha=0.2):
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix
    class_mat = tf.matmul(classes, tf.transpose(classes))
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = tf.reshape(dist_same - dist_diff + alpha,[-1])
    accuracy = tf.reduce_mean(tf.cast(loss<0, tf.float32))

    sorted_loss = tf.sort(loss, direction='DESCENDING')
    valid_loss = tf.boolean_mask(sorted_loss, tf.greater(sorted_loss,1e-16))
    nbof_kept_loss = tf.cast(tf.size(valid_loss),tf.float32)/(1+tf.exp(10*(accuracy-0.5)))

    nbof_kept_loss = tf.cast(tf.maximum(nbof_kept_loss,1),tf.int32)
    return tf.reduce_mean(sorted_loss[:nbof_kept_loss])

def binary_general_triplet_loss(embs, labels, minibatch_size, alpha=0.2):
    """
    NOTE: In order for this loss to work properly, it is prefered that 
    labels contains several repetitions. In other word:
    len(np.unique(labels))!=len(labels)
    """
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix, inner product of the classes
    class_mat = tf.matmul(classes, tf.transpose(classes))
    # Create mask for the same and different classes
    # The class matrix is symetric, so only the upper/lower triangular matrix
    # is important.
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = dist_same - dist_diff + alpha
    return tf.reduce_mean(tf.cast(tf.greater(loss,1e-16),tf.float32))

def hard_general_triplet_loss(embs, labels, minibatch_size, alpha=0.2):
    """
    WARNING: THIS IS NOT WORKING!
    NOTE: In order for this loss to work properly, it is prefered that 
    labels contains several repetitions. In other word:
    len(np.unique(labels))!=len(labels)
    """
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix, inner product of the classes
    class_mat = tf.matmul(classes, tf.transpose(classes))
    # Create mask for the same and different classes
    # The class matrix is symetric, so only the upper/lower triangular matrix
    # is important.
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = dist_same - dist_diff + alpha
    return tf.reduce_max(loss)
    # nbof_valid_triplets = tf.reduce_sum(tf.cast(tf.greater(loss,1e-16),tf.float32))
    # return tf.reduce_sum(tf.nn.relu(loss))/(nbof_valid_triplets+1e-16)

def focal_general_triplet_loss(embs, labels, minibatch_size, alpha=0.2):
    """
    NOTE: In order for this loss to work properly, it is prefered that 
    labels contains several repetitions. In other word:
    len(np.unique(labels))!=len(labels)
    """
    classes = tf.one_hot(labels,depth=minibatch_size)
    # Classes matrix, inner product of the classes
    class_mat = tf.matmul(classes, tf.transpose(classes))
    # Create mask for the same and different classes
    # The class matrix is symetric, so only the upper/lower triangular matrix
    # is important.
    mask_same = tf.matrix_band_part(class_mat - tf.matrix_band_part(class_mat,0,0), 0,-1)
    mask_same = tf.reshape(mask_same,[-1])
    mask_diff = tf.matrix_band_part(1-class_mat, 0,-1)
    mask_diff = tf.reshape(mask_diff,[-1])

    # Predictions matrix, inner product of the predictions
    tiled_emb = tf.expand_dims(embs,0)
    dist_mat = tf.reduce_sum(tf.square(tiled_emb-tf.transpose(tiled_emb,(1,0,2))),axis=-1)
    dist_mat = tf.reshape(dist_mat,[-1])

    # Distances and loss
    dist_same = tf.expand_dims(tf.boolean_mask(dist_mat,mask_same),0)
    dist_diff = tf.expand_dims(tf.boolean_mask(dist_mat,mask_diff),1)
    loss = dist_same - dist_diff + alpha
    eps = 1e-16
    rectified_loss = tf.clip_by_value(loss/(2+alpha+eps),eps,1-eps)
    rectified_loss = 1/(1+tf.exp(-10*(rectified_loss-0.6)))
    p = 1-rectified_loss
    new_loss = - (tf.square(p)*tf.log(p))
    return tf.reduce_mean(new_loss)

#############################################################################
# Softmax losses
#############################################################################
# https://github.com/auroua/InsightFace_TF/
# https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py
def arcface_loss(embedding, labels, out_num, regularizer_rate=0., w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.compat.v1.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding = tf.compat.v1.nn.l2_normalize(embedding, axis=1, name='norm_embedding')
        weights = tf.get_variable(
            name='embedding_weights',
            shape=(embedding.get_shape().as_list()[-1], out_num),
            initializer=w_init,
            regularizer=tf.keras.regularizers.l2(regularizer_rate),
            dtype=tf.float32)
        weights = tf.compat.v1.nn.l2_normalize(weights, axis=0, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.compat.v1.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.compat.v1.square(cos_t, name='cos_2')
        sin_t2 = tf.compat.v1.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.compat.v1.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.compat.v1.subtract(tf.compat.v1.multiply(cos_t, cos_m, name='cos_txcos_m'), tf.compat.v1.multiply(sin_t, sin_m, name='sin_txsin_m'), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.compat.v1.cast(tf.compat.v1.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.compat.v1.where(cond, cos_mt, keep_val)

        mask = tf.compat.v1.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask)
        inv_mask = tf.compat.v1.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.compat.v1.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.compat.v1.add(tf.compat.v1.multiply(s_cos_t, inv_mask, name='s_cos_txinv_mask'), tf.compat.v1.multiply(cos_mt_temp, mask, name='cos_mt_tempxmask'), name='arcface_loss_output')
    return output

def cosineface_losses(embedding, labels, out_num, regularizer_rate=0., s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(
            name='embedding_weights',
            shape=(embedding.get_shape().as_list()[-1], out_num),
            regularizer=tf.keras.regularizers.l2(regularizer_rate),
            dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output

#----------------------------------------------------------------------------
# Test

def test_arcface_loss():
    embedding = tf.random.normal(dtype=tf.float32, shape=[32,64])
    labels = tf.constant(np.arange(288))
    output = arcface_loss(embedding, labels, 10575)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        print(sess.run(output).shape)

def test_general_triplet_loss():
    # batch_size = 32
    embs = tf.random.normal(dtype=tf.float32, shape=[32,64])
    embedding = tf.nn.l2_normalize(embs, axis=-1)
    # embedding = embs*tf.rsqrt(tf.reduce_sum(tf.square(embs), axis=-1, keepdims=True))
    labels = tf.constant(np.concatenate([[i]*j for i,j in enumerate([2,4,2,6,8,6,4])]))
    output = general_triplet_accuracy(embedding,labels,32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for _ in range(10):
            out = sess.run(output)
            print(out)
        print(out.shape)

if __name__=='__main__':
    # test_arcface_loss()
    test_general_triplet_loss()

#----------------------------------------------------------------------------
