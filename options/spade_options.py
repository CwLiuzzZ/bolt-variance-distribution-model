from pickle import FALSE
from .base_options import BaseOptions

class SpadeOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--data_format', type=str, default='Yolo', help='# choose dataset for yolo or for test')
        self.parser.add_argument('--data_annotation_path', type=str, default='./dataset/defect/train/spade_train.txt', help='# choose data annotation for dataloader')
        self.parser.add_argument('--DataParallel', action='store_true', help='# use torch DataParallel')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument('--generate_noise', action='store_true', default=False, help='if specified, generate noise images in training')
        self.parser.add_argument('--noise_repeat_num', type=int, default=2, help='the number of noise images generated for each training image')
        self.parser.add_argument('--train_rgb', action='store_true', default=False, help="choose training mode")
        # self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        # self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        # self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")     

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')       
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = False

