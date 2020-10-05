import numpy as np

def define_batch(filenames, labels, minibatch_size, max_img_per_class=10):
    img_batch = np.empty((0))
    lab_batch = np.empty((0))
    classes = np.unique(labels)
    while len(lab_batch)<minibatch_size:
        # Select a class
        crt_class = np.random.choice(classes)
        # Check if not in selected classes
        if len(lab_batch)==0 or crt_class not in lab_batch:
            keep = np.equal(labels,crt_class)
            lab_batch = np.append(lab_batch, labels[keep][:max_img_per_class], axis=0)
            img_batch = np.append(img_batch, filenames[keep][:max_img_per_class], axis=0)
    return img_batch[:minibatch_size], lab_batch[:minibatch_size]

# def generate_random_triplets(filenames, labels, minibatch_size):
#     assert minibatch_size%3 == 0
#     # Select classes
#     classes = np.unique(labels)
#     selected_filenames = np.empty((minibatch_size),filenames.dtype)
#     selected_labels = np.empty((minibatch_size), labels.dtype)
#     for i in range(0,minibatch_size,3):
#         # Select a positive class
#         pos_class = np.random.choice(classes)
#         # Assert there are at least two pictures from that class
#         while len(np.sum(np.equal(labels,pos_class).astype(np.float32))) <= 1:
#             pos_class = np.random.choice(classes)
#         # Select a negative class
#         neg_class = np.random.choice(classes)
#         # Assert the classes are different
#         while neg_class==pos_class:
#             neg_class = np.random.choice(classes)
#         # Select images from the positive class
#         keep_pos = np.equal(labels,pos_class)
#         pos_labels = labels[keep_pos]
#         pos_filenames = filenames[keep_pos]
#         selected_pos_idx = np.random.choice(np.arange(len(pos_labels)), size=2, replace=False)
#         # Add the positive images
#         selected_filenames[[i,i+1]] = pos_filenames[selected_pos_idx]
#         selected_labels[[i,i+1]] = pos_labels[selected_pos_idx]
#         # Select images from the negative class
#         keep_neg = np.equal(labels, neg_class)

#     chosen_idx = np.random.choice(np.arange(len(filenames)), size=minibatch_size, replace=False)

#     return filenames[chosen_idx], labels

def define_triplets_batch(filenames,labels,nbof_triplet = 21 * 3):
    """
    Generates offline soft triplet.
    Given a list of file names of pictures, their specific label and
    a number of triplet images, returns an array of triplet of images
    and their specific labels.

    Args:
     - filenames: array of strings. List of file names of the pictures. 
     - labels: array of integers.
     - nbof_triplet: integer. Has to be a multiple of 3.
     
     Returns:
     - triplet_train: array of pictures --> a 4D array. 
     - y_triplet: array of integers of same dimension as the first
     dimension of triplet_train. Contains the labels of the pictures.
    """
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0,nbof_triplet,3):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classAP)
        keep_classAP = filenames[keep]
        while len(keep_classAP)<=1:
            classAP = classes[np.random.randint(len(classes))]
            keep = np.equal(labels,classAP)
            keep_classAP = filenames[keep]
        keep_classAP_idx = labels[keep]
        idx_image1 = np.random.randint(len(keep_classAP))
        idx_image2 = np.random.randint(len(keep_classAP))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(keep_classAP))

        triplet_train += [keep_classAP[idx_image1]]
        triplet_train += [keep_classAP[idx_image2]]
        y_triplet[i] = keep_classAP_idx[idx_image1]
        y_triplet[i+1] = keep_classAP_idx[idx_image2]
        # Pick a class for the negative picture
        classN = classes[np.random.randint(len(classes))]
        while classN==classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classN)
        keep_classN = filenames[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train += [keep_classN[idx_image3]]
        y_triplet[i+2] = keep_classN_idx[idx_image3]
        
    return triplet_train, y_triplet

def define_adaptive_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    Generates hard triplet for offline selection. It will consider the whole dataset.
    This function will also return the predicted values.
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
    Returns:
        -triplets
        -y_triplets: labels of the triplets
        -pred_triplets: predicted embeddings of the triplets
    """
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    pred_triplets = np.empty((nbof_triplet,predict.shape[-1]))
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                j += 1
                if j == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        pred_triplets[i] = predict[keep][idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        pred_triplets[i+1] = predict[keep][idx_image2]
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)

        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]
        pred_triplets[i+2] = predict[not_keep][idx_image3]

    return np.array(triplets), y_triplets, pred_triplets

def online_adaptive_hard_image_generator(
    filenames,                      # Absolute path of the images
    labels,                         # Labels of the images
    sess,                           # Tf session
    image_jpeg, images, training,   # Tf placeholder
    image_decoded, emb,             # Tf variables
    accuracy,                       # Current accuracy of the model
    batch_size      =63,            # Batch size (has to be a multiple of 3 for dogfacenet)
    nbof_subclasses =10):           # Number of subclasses from which the triplets will be selected
    """
    Generator to select online hard triplets for training.
    Include an adaptive control on the number of hard triplets included during the training.
    """

    # Proportion of hard triplets in the generated batch
    # hard_triplet_ratio = max(0,1.2/(1+np.exp(-10*accuracy+5.3))-0.19)
    # hard_triplet_ratio = max(0,1.2/(1+np.exp(-20*(accuracy-0.77)))-0.19)
    hard_triplet_ratio = 1/(1+np.exp(-20*(accuracy-0.77)))
    # hard_triplet_ratio = np.exp(-accuracy * 10 / batch_size)

    if np.isnan(hard_triplet_ratio):
        hard_triplet_ratio = 0
    nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)

    # Select a certain amount of subclasses
    classes = np.unique(labels)
    # In order to limit the number of computation for prediction,
    # we will not computes nbof_subclasses predictions for the hard triplets generation,
    # but int(nbof_subclasses*hard_triplet_ratio)+2, which means that the higher the
    # accuracy is the more prediction are going to be computed.
    subclasses = np.random.choice(classes,size=int(nbof_subclasses*hard_triplet_ratio)+2,replace=False)
    
    keep_classes = np.equal(labels,subclasses[0])
    for i in range(1,len(subclasses)):
        keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
    subfilenames = filenames[keep_classes]
    sublabels = labels[keep_classes]

    images_ = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in subfilenames]
    predict = np.concatenate([sess.run(emb, feed_dict={images:images_[i:i+batch_size], training:False}) for i in range(0,len(images_),batch_size)], axis=0)
    # predict = model.predict_generator(predict_generator(subfilenames, 32),
    #                                   steps=int(np.ceil(len(subfilenames)/32)))

    # f_triplet_hard, y_triplet_hard, predict_hard = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, nbof_hard_triplets*3, use_neg=True, use_pos=True)
    # f_triplet_soft, y_triplet_soft, predict_soft = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size-nbof_hard_triplets*3, use_neg=False, use_pos=False)
    f_triplet_hard, y_triplet_hard, _ = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, nbof_hard_triplets*3, use_neg=True, use_pos=True)
    f_triplet_soft, y_triplet_soft, _ = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size-nbof_hard_triplets*3, use_neg=False, use_pos=False)

    f_triplet = np.append(f_triplet_hard,f_triplet_soft)
    y_triplet = np.append(y_triplet_hard,y_triplet_soft)

    # predict = np.append(predict_hard, predict_soft, axis=0)


    # Potential modif for different losses: re-labels the dataset from 0 to nbof_subclasses
    # dict_subclass = {subclasses[i]:i for i in range(nbof_subclasses)}
    # ridx_y_triplet = [dict_subclass[y_triplet[i]] for i in range(len(y_triplet))]
    i_triplet = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in f_triplet]
    return i_triplet, y_triplet