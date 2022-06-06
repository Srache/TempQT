import PIL.Image
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import torchvision
from torchvision import transforms
import config
from openpyxl import load_workbook


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        # min-max normalization
        labels = (labels - labels.min()) / (labels.max() - labels.min())
        # labels = labels / 100

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']


        refname.sort()
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
        self.samples = sample

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        # min-max normalization
        labels = (labels - labels.min()) / (labels.max() - labels.min())
        # labels = labels / 100

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath,'.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        refname.sort(reverse=True)
        # refnames_all.sort()

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomCrop(size=224),
        #     torchvision.transforms.ToTensor(),
        # ])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                # mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos = float(row['MOS_zscore'])
                mos_all.append(mos)

        mos_all = np.array(mos_all)
        mos_all = (mos_all - mos_all.min()) / (mos_all.max() - mos_all.min())

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'labels_image.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'].split('/')[1])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'FLIVE', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
     
        

class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)

        # min-max normalization
        labels = (labels - labels.min()) / (labels.max() - labels.min())
        # labels = labels / 100

        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Kadid10k(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.png.PNG')
        # txtpath = os.path.join(root, 'dmos.txt')
        # fh = open(txtpath, 'r')

        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row['dist_img'])
                refnames_all.append(row['ref_img'][1:3])

                mos = np.array(float(row['dmos'])).astype(np.float32)
                target.append(mos)

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]),
                                   os.path.join(root, 'reference_images',
                                                imgnames[item].split('_', 1)[0] + '.' + imgnames[item].split('.', 1)[1]),
                                   labels[item]
                                   ))

        self.samples = sample


        # self.transform2 = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize([config.PATCH_SIZE, config.PATCH_SIZE]),
		# 	torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                      std=(0.229, 0.224, 0.225)),
        # ])

        self.transform_g = transforms.Compose([
            transforms.Resize([config.PATCH_SIZE, config.PATCH_SIZE]),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])

        self.transform_rgb = transforms.Compose([
            transforms.Resize([config.PATCH_SIZE, config.PATCH_SIZE]),
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path_dist, path_ref, target = self.samples[index]


        path_ref = './I01.png'
        path_dist = './I01_24_04.png'

        sample_dist = pil_loader(path_dist)
        sample_ref = pil_loader(path_ref)

        dist_rgb = self.transform_rgb(sample_dist)
        dist_g = self.transform_g(sample_dist)
        ref_g = self.transform_g(sample_ref)
        diff = dist_g - ref_g
        diff1 = ref_g - dist_g
        diff_f = (abs(diff1) + abs(diff)) / 2
        return dist_rgb, dist_g, ref_g, diff_f

        #
        # import matplotlib.pyplot as plt
        #
        # ref = ref_g.permute(1, 2, 0)
        # dist = dist_g.permute(1, 2, 0)
        # diff = diff.permute(1, 2, 0)
        # diff1 = diff1.permute(1, 2, 0)
        # diff_f = diff_f.permute(1, 2, 0)
        # fig = plt.figure()
        # ax = fig.add_subplot(231)
        # ax.set_title('Ref')
        # plt.axis('off')
        # im = ax.imshow(ref)
        #
        # ax = fig.add_subplot(232)
        # ax.set_title('Dist')
        # plt.axis('off')
        # im = ax.imshow(dist)
        #
        # ax = fig.add_subplot(233)
        # ax.set_title('D-R')
        # plt.axis('off')
        # im = ax.imshow(diff)
        #
        # ax = fig.add_subplot(234)
        # ax.set_title('R-D')
        # plt.axis('off')
        # im = ax.imshow(diff1)
        #
        # ax = fig.add_subplot(235)
        # ax.set_title('Avg')
        # plt.axis('off')
        # im = ax.imshow(diff_f)
        #
        # ax = fig.add_subplot(236)
        # ax.set_title('Rec')
        # im = ax.imshow(dist-diff_f)
        # plt.axis('off')
        #
        # fig.tight_layout()
        # plt.savefig('oem.png', dpi=600, bbox_inches='tight')
        # plt.show()
        # return dist_g, diff

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    import config
    from data_loader import DataLoader

    folder_path = {
        'live':     config.DATA_PATH['live'],
        'csiq':     config.DATA_PATH['csiq'],
        'tid2013':  config.DATA_PATH['tid2013'],
        'kadid10k': config.DATA_PATH['kadid10k'],
        'livec':    config.DATA_PATH['livec'],
        'koniq':    config.DATA_PATH['koniq'],
        # 'fblive':    config.data_path['fblive'],
        }

    img_num = {
        'live':     list(range(0, 29)),
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'livec':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        # 'fblive':   list(range(0, 39810)),
        }

    dataset = 'live'


    total_num_images = img_num[dataset]

    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]


    dataloader_train = DataLoader(dataset,
                              folder_path[dataset],
                              train_index,
                              config.PATCH_SIZE,
                              config.TRAIN_PATCH_NUM,
                              config.BATCH_SIZE,
                              istrain=True).get_data()

    dataloader_test = DataLoader(dataset,
                             folder_path[dataset],
                             test_index,
                             config.PATCH_SIZE,
                             config.TEST_PATCH_NUM,
                             istrain=False).get_data()

    for idx, (_, _) in enumerate(dataloader_train):
        print(_.shape)
        import sys
        sys.exit()
