import os
import json
import time
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import iml_utils.training_scripts.utils.misc as misc

from iml_utils.registry import MODELS, POSTFUNCS
from iml_utils.datasets import ManiDataset, JsonDataset

from iml_utils.evaluation import PixelF1, ImageF1

from iml_utils.training_scripts.tester import test_one_epoch
from bca_iml import BCAIML

# robustness wrappers
from iml_utils.transforms.robustness_wrapper import (
    GaussianBlurWrapper,
    GaussianNoiseWrapper,
    JpegCompressionWrapper
)

def get_args_parser():
    parser = argparse.ArgumentParser('Robustness test Launch!', add_help=True)
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
    parser.add_argument('--test_data_path', default='/data/ataset/CASIA1.0', type=str,
                        help='test dataset path, should be our JsonDataset or Manidataset format. Details are in readme.md')

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

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=42, type=int)
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
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0
    

    model = MODELS.get(args.model)
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)

    robustness_list = [
            GaussianBlurWrapper([0, 3, 7, 11, 15, 19, 23]),
            GaussianNoiseWrapper([3, 7, 11, 15, 19, 23]), 
            JpegCompressionWrapper([50, 60, 70, 80, 90, 100])
    ]
    
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
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
    
    for attack_wrapper in robustness_list:
        for attack_param, attack_transform in attack_wrapper:
            args.full_log_dir = os.path.join(args.log_dir, str(attack_transform))
            
            
            if global_rank == 0 and args.full_log_dir is not None:
                os.makedirs(args.full_log_dir, exist_ok=True)
                log_writer = SummaryWriter(log_dir=args.full_log_dir)
            else:
                log_writer = None
        
            # ---- dataset with crop augmentation ----
            if os.path.isdir(args.test_data_path):
                dataset_test = ManiDataset(
                    args.test_data_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=attack_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function
                )
            else:
                dataset_test = JsonDataset(
                    args.test_data_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=attack_transform,
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

            print(f"Start testing on {attack_wrapper}! ")
            chkpt_list = os.listdir(args.checkpoint_path)
            chkpt_pair = [(int(chkpt.split('-')[1].split('.')[0]), chkpt) for chkpt in chkpt_list if
                          chkpt.endswith(".pth")]
            chkpt_pair.sort(key=lambda x: x[0])
            for epoch, chkpt_dir in chkpt_pair:
                ckpt = os.path.join(args.checkpoint_path, chkpt_dir)
                if ckpt.endswith(".pth"):
                    print("Loading checkpoint: %s" % chkpt_dir)
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
        print(f'Testing on transforme {attack_transform} takes {local_time_str}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
