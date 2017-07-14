class ModelInfo(object):
    def __init__(self, **options):
        self.data_url = options.get('data_url')
        self.bottleneck_tensor_name = options.get('bottleneck_tensor_name')
        self.bottleneck_tensor_size = options.get('bottleneck_tensor_size')
        self.input_width = options.get('input_width')
        self.input_height = options.get('input_height')
        self.input_depth = options.get('input_depth')
        self.resized_input_tensor_name = options.get('resized_input_tensor_name')
        self.model_file_name = options.get('model_file_name')
        self.input_mean = options.get('input_mean')
        self.input_std = options.get('input_std')
