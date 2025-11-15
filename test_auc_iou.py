import os
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import iml_utils.training_scripts.utils.misc as misc

from iml_utils.registry import MODELS, POSTFUNCS
from iml_utils.datasets import ManiDataset, JsonDataset
from iml_utils.transforms import get_albu_transforms
from iml_utils.evaluation import PixelF1

from iml_utils.training_scripts.tester import test_one_epoch

from bca_iml import BCAIML


def get_args_parser():
    parser = argparse.ArgumentParser('testing launch!', add_help=True)
    # Model name
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')

    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--test_data_json', default='/data/dataset/CASIA1.0', type=str,
                        help='test dataset json, should be a json file contains many datasets. Details are in readme.md')

    parser.add_argument('--checkpoint_path', default = None, type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')

    parser.add_argument('--output_dir', default='./test_output/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./test_output/output_dir',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args, remaining_args = parser.parse_known_args()

    model_class = MODELS.get(args.model)

    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    test_transform = get_albu_transforms('test')

    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)
    
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0
    
    # Init model with registry
    model = MODELS.get(args.model)
    
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
        
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)

    evaluator_list = [
        PixelAUC(threshold=0.5, mode="origin"),  
        PixelIOU(threshold=0.5, mode="origin")  
    ]
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    
    # Start go through each datasets:
    for dataset_name, dataset_path in test_dataset_json.items():
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)

        if global_rank == 0 and args.full_log_dir is not None:
            os.makedirs(args.full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.full_log_dir)
        else:
            log_writer = None
        
        # ---- dataset with crop augmentation ----
        if os.path.isdir(dataset_path):
            dataset_test = ManiDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )

        else:
            dataset_test = JsonDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        # ------------------------------------
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))
        
        # Sampler
        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=False,
                drop_last=True
            )
            print("Sampler_test = %s" % str(sampler_test))
        else:
            sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        print(f"Start testing on {dataset_name}! ")
        chkpt_list = os.listdir(args.checkpoint_path)
        print(chkpt_list)
        chkpt_pair = [(int(chkpt.split('-')[1].split('.')[0]) , chkpt) for chkpt in chkpt_list if chkpt.endswith(".pth")]
        chkpt_pair.sort(key=lambda x: x[0])
        print( "sorted checkpoint pairs in the ckpt dir: ",chkpt_pair)
        for epoch , chkpt_dir in chkpt_pair:
            if chkpt_dir.endswith(".pth"):
                print("Loading checkpoint: %s" % chkpt_dir)
                ckpt = os.path.join(args.checkpoint_path, chkpt_dir)
                ckpt = torch.load(ckpt, map_location='cuda')
                model.module.load_state_dict(ckpt['model'], strict=False)
                test_stats = test_one_epoch(
                    model=model,
                    data_loader=data_loader_test,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=epoch,
                    log_writer=log_writer,
                    args=args
                )
                log_stats = {
                    **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}

                if args.full_log_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        print(f'Testing on dataset {dataset_name} takes {local_time_str}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List , Any
from sklearn.metrics import roc_auc_score
# from iml_utils.evaluation import AbstractEvaluator
from iml_utils.evaluation.abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from iml_utils.training_scripts.utils import misc
from sklearn.metrics import roc_auc_score
    

class ImageAUCNoRemain(AbstractEvaluator):
    def __init__(self) -> None:
        self.name = "image-level AUC"
        self.desc = "image-level AUC"
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')
    
    def compute_auc(self, y_true, y_scores):
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        auc = torch.trapz(tpr, fpr)

        return auc.item()
    
    
    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        predict = predict_label.float().cuda()
        self.predict_label = torch.cat([self.predict_label, predict], dim=0)
        self.label = torch.cat([self.label, label], dim=0)
        self.cnt += torch.tensor(len(label), device='cuda')
        return None

    def epoch_update(self):
        # cnt = torch.tensor(self.cnt, dtype=torch.int64).cuda()
        cnt = self.cnt.clone().detach().cuda()
        t_gather_cnt = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather(t_gather_cnt, cnt)
        
        max_cnt = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[0].cuda()
        max_idx = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[1].cuda()
        if max_cnt > self.cnt:
            self.predict_label = torch.cat([self.predict_label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)
            self.label = torch.cat([self.label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)

        t_label = self.label.float().cuda()
        t_predict_label = self.predict_label.float().cuda()

        t_gather_predict_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
        t_gather_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
        dist.barrier()

        dist.all_gather(t_gather_label, t_label)
        
        dist.barrier()
        dist.all_gather(t_gather_predict_label, t_predict_label)

        final_predict_label = torch.cat([t_gather_predict_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()
        final_label = torch.cat([t_gather_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()

        final_predict_label = final_predict_label.view(-1)
        final_label = final_label.view(-1)
        AUC = self.compute_auc(final_label, final_predict_label)
        return AUC
    
    def recovery(self):
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')

class ImageAUC(AbstractEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__() 
        self.name = "image-level AUC"
        self.desc = "image-level AUC"
        self.threshold = threshold
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        self.world_size = misc.get_world_size()
        self.local_rank = misc.get_rank()

    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        self.predict.append(predict_label)
        self.label.append(label)
        return None
        
    def remain_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params
        self.remain_predict.append(predict_label)
        self.remain_label.append(label)
        return None

    def epoch_update(self):
        if len(self.predict) != 0:
            predict = torch.cat(self.predict, dim=0)
            label = torch.cat(self.label, dim=0)
            gather_predict_list = [torch.zeros_like(predict) for _ in range(self.world_size)]
            gather_label_list = [torch.zeros_like(label) for _ in range(self.world_size)]
            dist.all_gather(gather_predict_list, predict)
            dist.all_gather(gather_label_list, label)
            gather_predict = torch.cat(gather_predict_list, dim=0)
            gather_label = torch.cat(gather_label_list, dim=0) 
            if len(self.remain_predict) != 0:
                self.remain_predict = torch.cat(self.remain_predict, dim=0)
                self.remain_label = torch.cat(self.remain_label, dim=0)
                gather_predict = torch.cat([gather_predict, self.remain_predict], dim=0)
                gather_label = torch.cat([gather_label, self.remain_label], dim=0)
        else:
            if len(self.remain_predict) == 0:
                raise RuntimeError(f"No data to calculate {self.name}, please check the input data.")
            gather_predict = torch.cat(self.remain_predict, dim=0)
            gather_label = torch.cat(self.remain_label, dim=0)
        # calculate AUC
        auc = roc_auc_score(gather_label.cpu().numpy(), gather_predict.cpu().numpy())
        return auc
    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_predict = []
        self.remain_label = []
        return None
    
    

class PixelAUC(AbstractEvaluator):
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level AUC"
        self.desc = "pixel-level AUC"
        self.threshold = threshold
        self.mode = mode

    def Cal_AUC(self, y_true, y_scores, shape_mask=None):
        if shape_mask is not None:
            y_true = y_true * shape_mask
            y_scores = y_scores * shape_mask
        
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()
        if torch.sum(y_true) == 0:
            # raise "The mask is all 0, we can't calculate pixel-AUC under this situation, please utilize a test only containse manipulated images to calculate AUC."
            return 0.0

        if shape_mask is not None:
            valid_mask = shape_mask.flatten() > 0
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]

        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        auc = torch.trapz(tpr, fpr)

        return auc.item()
        
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)
        AUC_list = []
        if self.mode == "origin":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask))
        elif self.mode == "reverse":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask))
        elif self.mode == "double":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(max(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask), self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask)))
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        
        return torch.tensor(AUC_list)
    
    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):

        return None
    
    def recovery(self):
        return None



class PixelIOU(AbstractEvaluator):
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level IOU"
        self.desc = "pixel-level IOU"
        self.threshold = threshold
        self.mode = mode
    
    def Cal_IOU(self, predict, mask, shape_mask=None):
        if shape_mask is not None:
            predict = predict * shape_mask
            mask = mask * shape_mask
        
        predict = (predict > self.threshold).float().flatten(1)
        mask = mask.flatten(1)


        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou

    def Cal_IOU_2(self, predict, mask, shape_mask=None):
        predict = (predict > self.threshold).float().to(torch.int8)
        mask = mask.to(torch.int8)
        predict = 1 - predict
        mask = 1 - mask
        if shape_mask is not None:
            predict = predict * shape_mask.to(torch.int8)
            mask = mask * shape_mask.to(torch.int8)

        # Flatten the tensors
        predict = predict.flatten(1)
        mask = mask.flatten(1)
        print(predict.shape)
        # Compute intersection and union
        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou
    
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)
        if self.mode == "origin":
            IOU = self.Cal_IOU(predict, mask, shape_mask)
            # IOU2 = self.Cal_IOU_2(predict, mask, shape_mask)
        elif self.mode == "reverse":
            IOU = self.Cal_IOU(1 - predict, mask, shape_mask)
            # IOU2 = self.Cal_IOU_2(1 - predict, mask, shape_mask)
        elif self.mode == "double":
            normal_iou = self.Cal_IOU(predict, mask, shape_mask)
            reverse_iou = self.Cal_IOU(1 - predict, mask, shape_mask)
            IOU = torch.max(normal_iou, reverse_iou)
            # normal_iou2 = self.Cal_IOU_2(predict, mask, shape_mask)
            # reverse_iou2 = self.Cal_IOU_2(1 - predict, mask, shape_mask)
            # IOU2 = torch.max(normal_iou2, reverse_iou2)
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        return IOU
    def epoch_update(self):

        return None
    
    def recovery(self):
        return None

if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)