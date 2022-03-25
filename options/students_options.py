from .base_options import BaseOptions

class StudentsOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--max_epoch', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=400, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--data_format', type=str, default='Yolo', help='# choose dataset for Yolo or for test')
        self.parser.add_argument('--data_annotation_path', type=str, default='./dataset/defect/train/train.txt', help='# choose data annotation for train dataloader')
        self.parser.add_argument('--valid_data_annotation_path', type=str, default='./dataset/defect/train/valide.txt', help='# choose data annotation for valid dataloader')
        self.parser.add_argument('--DataParallel', action='store_true', help='# use torch DataParallel')
        # size = 17, effective if we are looking for small size anomalies
        # size = 33, effective if we are looking for medium size anomalies
        # size = 65, effective if we are looking for big size anomalies
        self.parser.add_argument('--use_whole_feature', type=bool, default=True,  help='use whole feature map or keypoint feature only')
        self.parser.add_argument('--anomaly_threshold', type=float, default=0.5,help='over which in score map will be seen as anomaly')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--generate_noise', action='store_true', default=False, help='if specified, generate noise images in training')
        self.parser.add_argument('--noise_repeat_num', type=int, default=4, help='the number of noise images generated for each training image')
        self.parser.add_argument('--train_rgb', action='store_true', default=False, help="choose training mode")
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=4, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=5.0, help='weight for feature matching loss')             
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = False
