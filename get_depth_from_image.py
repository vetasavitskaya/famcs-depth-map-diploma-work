import numpy as np
import mxnet as mx
import gluoncv
import PIL.Image as pil
from matplotlib import colors, cm, pyplot as plt
from mxnet.gluon.data.vision import transforms


def get_depth_from_single_image(frame):
    cpu_context = mx.cpu(0)
    original_width, original_height = frame.size
    kitti_model_resolution = (640, 192)
    # изменяем размер изображения, чтобы он имел тот же размер, что и предварительно обученная модель,
    frame = frame.resize(kitti_model_resolution, pil.LANCZOS)
    frame = transforms.ToTensor()(mx.nd.array(frame)).expand_dims(0).as_in_context(context=cpu_context)
    # получаем предварительно обученную модель
    model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_mono_stereo_640x192', pretrained_base=False,
                                        ctx=cpu_context, pretrained=True)
    prediction = model.predict(frame)
    disparity_map_predictions = prediction[("disp", 0)]
    disparity_map_predictions_resized = mx.nd.contrib.BilinearResize2D(disparity_map_predictions,
                                                                       height=original_height, width=original_width)
    disparity_array = disparity_map_predictions_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
    percentile = np.percentile(disparity_array, 100)
    # scale_converter будет масштабировать данные  в интервале от disparity_array.min() до percentile
    scale_converter = colors.Normalize(vmin=disparity_array.min(), vmax=percentile)
    # нормализация данных перед возвратом цветов RGBA из заданной палитры
    scalar_to_rgb_converter = cm.ScalarMappable(norm=scale_converter)
    normalized_rgba_array = (scalar_to_rgb_converter.to_rgba(disparity_array)[:, :, :3] * 255).astype(np.uint8)
    result_image = pil.fromarray(normalized_rgba_array)
    plt.imshow(result_image)
    plt.show()
