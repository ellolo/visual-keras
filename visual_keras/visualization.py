import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import _Pooling2D
from visual_keras import saliency
from visual_keras.utils import preproc_image, get_layers_output, visual_grid_3d_multichannel, visual_grid_4d, \
    visual_grid_4d_rgb, remove_last_layer_activation
from visual_keras.activation_maximization import image_filter_maximizer
from visual_keras.knn import compute_knn


def viz_conv_filters(model, layer_names, fig_size=None):
    """
    Visualizes the weights of convolutional filters using matplotlib.
    If a filter has 3 input channels, the weights are visualized as rgb images, otherwise as a N grayscale images,
    where N is the number of channels.

    Parameters
    ----------
    model : keras.Model
        Keras model
    layer_names : list
        List of conv layer names to be visualized. An image will be created for each layer
    fig_size : tuple
        A tuple (width, height) of the displayed figure in pixels. If None size will be inferred (default is None)

    Raises
    ------
    ValueError
        if a layer name does not refer to a layer with class Conv2
    """

    if fig_size is None:
        fig_size = (20, 20 * len(layer_names))
    fig = plt.figure(figsize=fig_size)
    i = 1
    for l in layer_names:
        layer = model.get_layer(l)
        if not isinstance(layer, Conv2D):
            raise ValueError("Layer {} is not 2d convolutional".format(l))
        weights = layer.get_weights()[0]
        channels = weights.shape[2]
        if channels != 3:
            cmap = "gray"
            ylab = "filters"
            xlab = "input channels"
        else:
            cmap = "viridis"
            ylab = ""
            xlab = ""
        grid = visual_grid_4d(weights, padding=1)
        sub = fig.add_subplot(len(layer_names), 1, i, xticks=[], yticks=[], ylabel=ylab, xlabel=xlab)
        sub.set_title(l)
        sub.imshow(grid, cmap=cmap)
        i += 1
    fig.show()


def viz_activations(image_path, model, layer_names, img_size=(224, 224), f_preproc=None, fig_size=None):
    """
    Visualizes the output of convolutional abnd pooling layers as activated by an input image using matplotlib.

    Parameters
    ----------
    image_path : String
        Path of the image for which to compute activations
    model : keras.Model
        Keras model
    layer_names: list
        List of conv/pooling layer names to be visualized. An image will be created for each layer
    img_size : tuple
        tuple (height, width) to which the input image will be rescaled before running in the model (default
        is (224, 224))
    f_preproc : function
        preprocessing function to be applied to the image before running in the model (default is None)
    fig_size: tuple
        tuple (width, height) of the output figure in pixels. If None size will be inferred (default is None)
    """

    if fig_size is None:
        fig_size = (10, 10 * len(layer_names))
    x = preproc_image(image_path, f_preproc=f_preproc, img_size=img_size)

    # check if layers are either convs or pools
    for l in layer_names:
        layer = model.get_layer(l)
        if not isinstance(layer, Conv2D) and not isinstance(layer, _Pooling2D):
            raise ValueError("Layer {} is not 2d convolutional or 2d pooling".format(l))

    outputs = get_layers_output(x, model, layer_names)
    fig = plt.figure(figsize=fig_size)

    for i in range(len(layer_names)):
        l_out = outputs[i][0]
        grid = visual_grid_3d_multichannel(l_out, padding=1, pixel_transform="standardized")
        sub = fig.add_subplot(len(layer_names), 1, i + 1, xticks=[], yticks=[])
        sub.set_title(layer_names[i])
        sub.imshow(grid)
        i += 1
    fig.show()


def viz_nearest_neighbors(q_paths, i_paths, model, layer_name, k=8, img_size=(224, 224, 3), f_preproc=None,
                          fig_size=(20, 20)):
    """
    Given an input list of query images, returns for each query image the top-k most similar images from the list of
    index images. Similarity is computed on the vectors outputted by a given layer of the model's network.
    Euclidean distance is used as similarity measure

    Parameters
    ----------
    q_paths : list
        List of paths for the query images
    i_paths : list
        List of paths for the index images
    model: keras.Model
        Keras model for the network
    layer_name : String
        name of the network layer whose output will be used to compute similarity
    k : int
        Number of most similar index images to be returned for each query image (default is 8)
    img_size : tuple
        Image size (height, width, channels) to be displayed (default is (224, 224, 3))
    f_preproc : function
        preprocessing function to be applied to the image before running in the model (default is None)
    fig_size: tuple
        tuple (width, height) of the output figure in pixels. If None size will be inferred (default is (20, 20))
    """

    out_dim = reduce(lambda a, b: a * b, model.get_layer(layer_name).output_shape[1:])
    q_vecs = np.zeros((len(q_paths), out_dim))
    i_vecs = np.zeros((len(i_paths), out_dim))
    q_imgs = []

    # get flattened layer output vector for query images
    for i in range(len(q_paths)):
        img = image.load_img(q_paths[i], target_size=img_size[:2])
        q_imgs.append(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if f_preproc is not None:
            x = f_preproc(x)
        vec = get_layers_output(x, model, [layer_name])[0][0].flatten()
        q_vecs[i] = vec

    # get flattened layer output vector for index images
    for i in range(len(i_paths)):
        x = preproc_image(i_paths[i], img_size=img_size, f_preproc=f_preproc)
        i_vecs[i] = get_layers_output(x, model, [layer_name])[0][0].flatten()

    # compute top k most similar index images for each query image
    top_k = compute_knn(q_vecs, i_vecs, k=k)
    images = np.zeros((img_size[0], img_size[1], img_size[2], len(q_imgs) * (k + 1)))

    # create visual grid for visualization and plot
    ct = 0
    for i in range(len(q_imgs)):
        images[:, :, :, ct] = q_imgs[i]
        ct += 1
        for j in range(k):
            img = image.load_img(i_paths[top_k[i][j]], target_size=img_size)
            images[:, :, :, ct] = img
            ct += 1
    grid = visual_grid_4d_rgb(images, padding=4, pixel_transform="normalized", dim=(len(q_imgs), k + 1))
    plt.figure(figsize=fig_size)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)
    plt.show()


def viz_class_score_maximizer(model, class_idx, img_size=(224, 224, 3), lrate=1., epochs=10, reg=None, disc_dead=False,
                              layer_name=None, remove_activation=True, fig_size=(10, 10)):
    """
    Visualizes the synthetic rgb image that maximizes the score of a given class, applying the method described in
    "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014

    Parameters
    ----------
    model : keras.Model
        Keras model
    class_idx : int
        Index of the target class in the last dense layer
    img_size : tuple
        Size of image to be created (height, width, channels)
    lrate : float
        learning rate for SGD (default is 25)
    epochs : int
        number of epochs for SGD (default is 10)
    reg : int
        Coefficient for L2 regularization. If None, no regularization is applied (default is None)
    disc_dead : boolean
        If True returns None when the activation is dead. i.e zero valued (default is False)
    layer_name : String
        Name of the last dense layer. If None the last layer of the model will be used (default is None)
    remove_activation : boolean
        If True (recommended) the activation of the last layer will be removed, giving optimal results (default is True)
    fig_size : tuple
        (width, height) of the output figure in pixel
    """

    if remove_activation:
        model = remove_last_layer_activation(model)

    if layer_name is None:
        layer_name = model.layers[-1].name

    imge = image_filter_maximizer(model, layer_name, class_idx, img_size, lrate=lrate, reg=reg, epochs=epochs,
                                  disc_dead=disc_dead)
    if disc_dead and imge is None:
        return
    imge = np.stack([imge], axis=3)  # shape format: (H, W, C, F)
    grid = visual_grid_4d_rgb(imge, pixel_transform="standardized")
    plt.figure(figsize=fig_size)
    plt.title("Class " + str(class_idx), fontsize=20)
    plt.imshow(grid)
    plt.show()


def viz_layer_maximizers(model, layer_name, img_size=(224, 224, 3), filters=None, lrate=1., epochs=10, reg=None,
                         disc_dead=False, fig_size=(20, 20)):
    """
    Visualizes the synthetic rgb images that maximize the filters of the given layer, applying the method described in
    "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014

    Parameters
    ----------
    model : keras.Model
        Keras model
    layer_name : String
        Name of the target layer
    img_size : tuple
        Size of image to be created (height, width, channels)
    filters: list
        Indexes of the filters in the target layer that the image will be displayed for
    lrate : float
        learning rate for SGD (default is 25)
    epochs : int
        number of epochs for SGD (default is 10)
    reg : int
        Coefficient for L2 regularization. If None, no regularization is applied (default is None)
    disc_dead : boolean
        If True returns None when the activation is dead. i.e zero valued (default is False)
    fig_size : tuple
        (width, height) of the output figure in pixel
    """

    if filters is None:
        num_filters = model.get_layer(layer_name).get_weights()[0].shape[-1]
        target_filters = range(num_filters)
    else:
        target_filters = filters
    images = list(map(lambda flt_id:
                      image_filter_maximizer(model, layer_name, flt_id, img_size, lrate=lrate,
                                             reg=reg, epochs=epochs, disc_dead=disc_dead),
                      target_filters))
    if disc_dead:
        images = list(filter(lambda img: img is not None, images))
    if len(images) != 0:
        images = np.stack(images, axis=3)  # shape format: (H, W, C, F)
        grid = visual_grid_4d_rgb(images, pixel_transform="standardized")
        plt.figure(figsize=fig_size)
        plt.title("Filter maximizers: " + layer_name, fontsize=20)
        plt.imshow(grid)
        plt.show()


def viz_saliency_map(image_path, model, class_idx, img_size=(224, 224, 3), f_preproc=None, variant="vanilla",
                     layer_name=None, spread=0.20, samples=50, multiply=False, blend=False, pair=False):
    """
    Visualizes the saliency map for an input image with respect to a specified class. As described in:
    "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014

    Parameters
    ----------
    image_path : string
        Path of the image
    model : keras.Model
        Keras model
    class_idx : int
        Index of the target class in the last dense layer
    img_size : tuple
        Tuple (height, width, channels) to which the input image will be rescaled before running in the model (default
        is (224, 224))
    f_preproc : function
        preprocessing function to be applied to the image before running in the model (default is None)
    variant: String
        Yype of saliency map to be applied. Possible values are:
            vanilla: as described in:
                "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
            smooth: as described in:
                "SmoothGrad: removing noise by adding noise"
            integated: as in:
                "Axiomatic Attribution for Deep Networks"
            gradcam: as described in:
                 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    layer_name : String
        Name of the layer for which saliency map will be visualized (only available for variants "vanilla" and
        "gradcam"). If None input layer will be used (default is None)
    spread : float
        controls the magnitude of the standard deviation of gaussian noise to be applied for variant "smooth"
        (suggested value >0.1 <0.2) (default is 0.2)
    samples : int
        number of saliency maps to compute and to average from (suggested value <50) for variants "smooth" and
        "integrated" (default is 50)
    multiply : boolean
        If True if will multiply the map by the input image (default is False)
    blend : boolean
        If true visualizes a blend of the target image and the salency map (default is False)
    pair : boolean
        If true visualizes the target image on the left and the salency map on the right (default is False)
    """

    x = preproc_image(image_path, img_size=img_size, f_preproc=f_preproc)

    if variant == "vanilla":
        provider = saliency.BaseSaliencyMap(model, layer_name=layer_name, multiply=multiply)
        heatmap = provider.get_map(x, class_idx)

    elif variant == "smooth":
        provider = saliency.SmoothGradMap(model, multiply=multiply)
        heatmap = provider.get_map(x, class_idx, samples=samples, spread=spread)

    elif variant == "integrated":
        provider = saliency.IntegratedGradientMap(model, multiply=multiply)
        heatmap = provider.get_map(x, class_idx, samples=samples)

    elif variant == "gradcam":
        provider = saliency.GradCamMap(model, layer_name=layer_name)
        heatmap = provider.get_map(x, class_idx)
    else:
        raise ValueError("Invalid variant: {}".format(variant))

    pil_img = Image.open(image_path)
    cm = plt.get_cmap('jet')
    pil_heatmap = Image.fromarray(np.uint8(cm(heatmap) * 255))
    pil_heatmap = pil_heatmap.resize((pil_img.size[0], pil_img.size[1]))
    if pair:
        paired_width = pil_img.size[0] * 2
        paired_height = pil_img.size[1]
        pil_paired = Image.new('RGB', (paired_width, paired_height))
        pil_paired.paste(pil_img, (0, 0))
        pil_paired.paste(pil_heatmap, (pil_img.size[0], 0))
        return pil_paired
    if blend:
        return Image.blend(pil_img.convert("RGBA"), pil_heatmap.convert("RGBA"), alpha=.5)

    else:
        return pil_heatmap
