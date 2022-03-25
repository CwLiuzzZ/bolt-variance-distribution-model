import numpy as np
import os
import ntpath
import time
from . import util
import torchvision.utils as vutils
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.win_size = opt.display_winsize
        self.name = opt.name

        self.win_size = 256
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        # print('Create directory %s...' % self.img_dir)
        util.mkdir(self.img_dir)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info  = f'{opt.name}\n'
        self.write_to_log_file(text=title + info)
        
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##              
    def write_to_log_file(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        if self.opt.display: # save images and display image
            win_index = 1
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch_%.3d_%s_%d.jpg' % (epoch, label, i))
                        # util.save_image(image_numpy[i], img_path)
                        self.vis.images(image_numpy[i], win=win_index, opts={'title': label})
                        win_index += 1
                else:                   
                    img_path = os.path.join(self.img_dir, 'epoch_%.3d_%s.jpg' % (epoch, label))
                    # util.save_image(image_numpy, img_path)
                    if len(image_numpy.shape) > 2:
                        image_numpy = np.transpose(image_numpy, (2, 0, 1))
                    self.vis.images(image_numpy, win=win_index, opts={'title': label})
                    win_index += 1

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, errors, counter_ratio, step):
        
        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        # print(f'[{epoch},{counter_ratio}]')
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
        Y = np.array(self.plot_data['Y'])
        if X.shape[1] == 1:
            X = X.squeeze(1)
            Y = Y.squeeze(1)
        self.vis.line(
            X=X, Y=Y,
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=10
        )        

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None: 
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=7
        )

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, iters, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        message = '   Loss: [%d/%d], Iters:%d ' % (epoch, self.opt.niter + self.opt.niter_decay, iters)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

                    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)

        message += 'min_err_g: %.3f' % best

        print(message)
        self.write_to_log_file(text=message)

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        # webpage.add_header(name)
        # ims = []
        # txts = []
        # links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

        #     ims.append(image_name)
        #     txts.append(label)
        #     links.append(image_name)
        # webpage.add_images(ims, txts, links, width=self.win_size)


class Visualizer_visdom():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info  = f'{opt.abnormal_class}, {opt.nz}, {opt.w_adv}, {opt.w_con}, {opt.w_lat}\n'
        self.write_to_log_file(text=title + info)


    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None: 
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def write_to_log_file(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)

        message += 'min_err_g: %.3f' % best

        print(message)
        self.write_to_log_file(text=message)

    def display_current_images(self, reals, noise, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        noise = self.normalize(noise.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(noise, win=2, opts={'title': 'Noise'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})

    def save_current_images(self, epoch, reals, noise, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(noise, '%s/noise.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_noise_%03d.png' %(self.img_dir, epoch+1), normalize=True)