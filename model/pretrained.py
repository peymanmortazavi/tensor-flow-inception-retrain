from model.info import ModelInfo


class Inception:

    @staticmethod
    def get_model_info(**options):
        return ModelInfo(
            data_url='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
            bottleneck_tensor_name='pool_3/_reshape:0',
            bottleneck_tensor_size=2048,
            input_width=299,
            input_height=299,
            input_depth=3,
            resized_input_tensor_name='Mul:0',
            model_file_name='classify_image_graph_def.pb',
            input_mean=128,
            input_std=128,
        )
