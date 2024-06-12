import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch import optim

from datasets.celeba import Celeba
from datasets.celeb_dfv2 import CelebDF
from datasets.lsun_bed import lsun_bed

from efficientnet_pytorch import EfficientNet
# from transformers import ViTModel
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from trainer import Trainer

from transform import Two_Path, get_augs
from consistency import Cons_LossL1, Cons_LossL2, Cons_LossCos

from utils import log_print

torch.multiprocessing.set_sharing_strategy("file_system")

def main(args):
    save_dir = os.path.join("ckpt",args.dataset, args.exp_name,args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    logfile = '{}/{}.log'.format(save_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    log_print("args: {}".format(args))

    # model
    if args.model_name == "efficientnetb0":
        model =  EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    elif args.model_name == "efficientnetb1":
        model =  EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
    elif args.model_name == "efficientnetb2":
        model =  EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
    elif args.model_name == "efficientnetb3":
        model =  EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
    elif args.model_name == "efficientnetb4":
        model =  EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    elif args.model_name == "efficientnetb5":
        model =  EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)
    elif args.model_name == "efficientnetb6":
        model =  EfficientNet.from_pretrained('efficientnet-b6', num_classes=2)
    elif args.model_name == "efficientnetb7":
        model =  EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    elif args.model_name == "vit_b_16":
        # model =  vit_b_16(weights=ViT_B_16_Weights.DEFAULT, num_classes=2)
        model =  vit_b_16(pretrained=False, num_classes=2)
        # model.load_state_dict(torch.load('./pretrained/vit_b_16-c867db91.pth'))
    elif args.model_name == "vit_b_32":
        model =  vit_b_16(pretrained=False, num_classes=2)
    elif args.model_name == "vit_l_16":
        model =  vit_b_16(pretrained=False, num_classes=2)
    elif args.model_name == "vit_l_32":
        model =  vit_b_16(pretrained=False, num_classes=2)
    elif args.model_name == "vit_h_14":
        model =  vit_b_16(pretrained=False, num_classes=2)
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        model = model.cuda()

    # transforms
    train_augs = get_augs(name=args.aug_name,norm=args.norm,size=args.size)
    if args.cons_loss != "None":
        train_augs = Two_Path(train_augs)
    log_print("train aug:{}".format(train_augs))
    test_augs = get_augs(name="None",norm=args.norm,size=args.size)
    log_print("test aug:{}".format(test_augs))

    # dataset
    if args.dataset == "celeba":
        train_dataset = Celeba(args.root,"train",train_augs, diffusion_type = 'ldm-dd')
        test_dataset = Celeba(args.root,"test",test_augs, diffusion_type = 'ldm-dd')
    elif args.dataset == "celebdf":
        train_dataset = CelebDF(args.root,"train",train_augs)
        test_dataset = CelebDF(args.root,"test",test_augs)
    elif args.dataset == "lsun_bed":
        train_dataset = lsun_bed(args.root,"train",train_augs, diffusion_type = args.diffusion_type)
        test_dataset = lsun_bed(args.root,"test",test_augs, diffusion_type = args.diffusion_type)
    elif args.dataset == "full":
        train_dataset = lsun_bed(args.root,"train",train_augs, diffusion_type = args.diffusion_type)
        test_dataset = lsun_bed(args.root,"test",test_augs, diffusion_type = args.diffusion_type)
    else:
        raise NotImplementedError

    log_print("len train dataset:{}".format(len(train_dataset)))
    log_print("len test dataset:{}".format(len(test_dataset)))
    # dataloader
    trainloader = DataLoader(train_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )
    testloader = DataLoader(test_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )

    if args.num_classes == 2:
        ce_weight = [args.real_weight, 1.0]
    else:
        raise NotImplementedError

    # Loss Function
    weight=torch.Tensor(ce_weight)
    if torch.cuda.is_available():
        weight = weight.cuda()
    loss_fn = nn.CrossEntropyLoss(weight)

    if args.cons_loss == "None":
        cons_loss_fn = None
    else:
        if args.cons_loss == "cons_loss":
            cons_loss_fn = Cons_LossCos()
        elif args.cons_loss == "L2_loss":
            cons_loss_fn = Cons_LossL2()
        elif args.cons_loss == "L1_loss":
            cons_loss_fn = Cons_LossL1()
        else:
            raise NotImplementedError
        log_print("consistency loss function: {}, rate:{}".format(cons_loss_fn, args.cons_loss_rate))

    # optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(),lr=args.lr)
    else:
        raise NotImplementedError
    log_print("optimizer: {}".format(optimizer))

    if args.load_model_path is not None:
        log_print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.load_model_path) # , map_location="cpu"
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_recond = {
            "acc": checkpoint['acc'],
            "auc": checkpoint['auc'],
            "epoch": checkpoint['epoch'],
        }
        start_epoch = checkpoint['epoch'] + 1
        log_print("start from best recode: {}".format(best_recond))
    else:
        best_recond={"acc":0,"auc":0,"epoch":-1,"tdr3":0,"tdr4":0}
        start_epoch = 0

    # trainer
    trainer = Trainer(
        model_name=args.model_name,
        train_loader=trainloader, 
        test_loader=testloader,
        model=model, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        consistency_fn=cons_loss_fn,
        consistency_rate=args.cons_loss_rate,
        log_interval=args.log_interval, 
        best_recond=best_recond,
        save_dir=save_dir,
        exp_name=args.exp_name,
        amp=args.amp)

    for epoch_idx in range(start_epoch,args.epochs):
        trainer.train_epoch(epoch_idx)
        trainer.test_epoch(epoch_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # dataset 
    arg('--num-classes', type=int, default=2)
    arg('--dataset', type=str, default='celeba') # lsun_bed, celebdf, celeba 
    arg('--root', type=str, default='data/celebahq')
    arg('--diffusion_type', type=str, default='ldm-dd') #ldm-dd, ldm-dd
    arg('--batch-size', type=int, default=16)
    arg('--num-workers', type=int, default=8)
    arg('--shuffle', type=bool, default=True)
    arg('--real-weight', type=float, default=4.0)
    # transforms
    arg('--aug-name', type=str, default="None") # None, RandomResizedCrop, RandomErasing, RanSelect
    arg('--norm', type=str, default="0.5")
    arg('--size', type=int, default=224)
    # optimizer
    arg('--optimizer', type=str, default="adam")
    arg('--lr', type=float, default=0.0002)
    arg('--exp_name', type=str, default='train')
    arg('--gpus', type=str, default='0')
    arg('--log-interval', type=int, default=100)
    arg("--epochs", type=int, default=20)
    arg("--load-model-path", type=str, default=None)
    arg("--model-name", type=str, default="efficientnetb0") # efficientnetb0-> efficientnetb7, vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
    arg("--amp", default=False, action='store_true')
    arg("--seed", type=int, default=3407)
    # consistency loss
    arg('--cons_loss', type=str, default="None") # cons_loss, L1_loss, L2_loss
    arg('--cons_loss_rate', type=float, default=1)


    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed) 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
