import os
import torch
import torch.utils.data as data
import basicsr.data.util as util

class Dataset_PillImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PillImage, self).__init__()
        self.opt = opt
        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt['dataroot_lq']
        self.train_size = opt['train_size']
        self.phase = opt.get('phase', 'train')
        self.use_flip = opt.get('use_flip', False)
        self.use_rot = opt.get('use_rot', False)

        # load image filenames
        self.gt_paths = sorted(util.glob_file_list(self.gt_root))
        self.lq_paths = sorted(util.glob_file_list(self.lq_root))

        assert len(self.gt_paths) == len(self.lq_paths), \
            f'Mismatched GT and LQ: {len(self.gt_paths)} vs {len(self.lq_paths)}'

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]

        gt = util.read_img_seq2([gt_path], self.train_size)[0]
        lq = util.read_img_seq2([lq_path], self.train_size)[0]

        if self.phase == 'train':
            imgs = [lq, gt]
            imgs = util.augment_torch(imgs, self.use_flip, self.use_rot)
            lq, gt = imgs

        return {
            'gt': gt,
            'lq': lq,
            'gt_path': gt_path,
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.gt_paths)