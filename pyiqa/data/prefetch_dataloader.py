import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


class DataLoaderX(torch.utils.data.DataLoader):
    """
    使用prefetch_generator包提供的数据预加载功能,需要安装prefetch_generator
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    @staticmethod
    def my_collate_fn(batch):
        """
        把一个batch内部的数据按照score进行降序排序，
        :param batch: 图像，score的列表 [[image_0, score_0], [image_1, score_1], ...]
        :return:
        """
        # print(type(batch[0]))
        # imgs = batch['img']
        # mos_labels = batch['mos_label']
        # imgs_path = batch["img_path"]
        #
        # rankbatch = [[img, mos, path] for img, mos, path in zip(imgs, mos_labels, imgs_path)]
        # rankbatch.sort(key=lambda x: x[1], reverse=True)
        #
        image_sequence = []
        score_sequence = []
        imgs_path_sequence = []
        batch.sort(key=lambda x: x['mos_label'], reverse=True)

        for item in batch:
            image_sequence.append(item['img'])
            score_sequence.append(item['mos_label'])
            imgs_path_sequence.append(item['img_path'])

        image_sequence = torch.utils.data.dataloader.default_collate(image_sequence)
        score_sequence = torch.utils.data.dataloader.default_collate(score_sequence)
        imgs_path_sequence = torch.utils.data.dataloader.default_collate(imgs_path_sequence)

        return {"img": image_sequence, "mos_label": score_sequence, "img_path": imgs_path_sequence}
