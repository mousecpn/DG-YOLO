# IRM
from DG_model import *
from utils.utils import *
from da_datasets import *
from utils.parse_config import *
from test import evaluate
from tensorboardX import SummaryWriter
import itertools
from terminaltables import AsciiTable
import math

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from domain_classifier import domain_cls, ReverseLayerF

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,
                        default="./weights/darknet53.conv.74",
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)
    writer = SummaryWriter(comment='dg-yolo-v3')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    d_classifier = domain_cls().to(device)
    loss_domain = torch.nn.BCELoss()
    model.apply(weights_init_normal)
    init_lr = 1e-3
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), d_classifier.parameters()), lr=init_lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    epoch_start = 0
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            all_state = torch.load(opt.pretrained_weights)
            model.load_state_dict(all_state['model'])
            optimizer.load_state_dict(all_state['optimizer'])
            epoch_start = all_state['epoch'] + 1
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )


    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(epoch_start,opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets,domain_labels) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            total_batches = len(dataloader) * opt.epochs
            len_dataloader = len(dataloader)
            p = float(batch_i + epoch * len_dataloader) / opt.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # IRM dummy_w
            scale = torch.tensor(1.).to(device).requires_grad_()

            imgs = Variable(imgs.to(device), requires_grad=False)
            targets = Variable(targets.to(device), requires_grad=False)
            domain_labels = Variable(domain_labels.to(device), requires_grad=False)

            loss, outputs,features,pen = model(imgs, targets, da=True, IRM=True, dummy_w=scale)
            
            mask = domain_labels[:,0] == 1
            mask = ~mask
            domain_labels = domain_labels[:, 1:]
            domain_labels = domain_labels[mask]

            # IRM
            gradient = torch.autograd.grad(loss, [scale], create_graph=True)[0]
            penalty = torch.sum(gradient ** 2) + pen

            # domain loss
            features = features[mask]
            domain_pred = d_classifier(features)
            
            loss_d = loss_domain(domain_pred,domain_labels)
            total_loss = loss + penalty + alpha * loss_d

            total_loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                writer.add_scalar('loss', loss.item(), batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nloss {loss.item()}"
            log_str += f"\npenalty loss {penalty.item()}"
            log_str += f"\ndomain loss {loss_d.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if (epoch % opt.evaluation_interval == 0) and epoch > 30:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                augment=False
            )
            _, _, AP_DG, _, _ = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                augment=True
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP_ori", AP.mean()),
                ("val_mAP_DG", AP_DG.mean()),
                ("val_f1", f1.mean()),
            ]
            writer.add_scalar('val_precision', precision.mean(), epoch)
            writer.add_scalar('val_recall', recall.mean(), epoch)
            writer.add_scalar('val_mAP_ori', AP.mean(), epoch)
            writer.add_scalar('val_mAP_DG', AP_DG.mean(), epoch)
            writer.add_scalar('val_f1', f1.mean(), epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                writer.add_scalar(class_names[c], AP[i], epoch)
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0 and (epoch>30):
            all_states = {'model':model.state_dict(),
                          "optimizer":optimizer.state_dict(),
                          'epoch':epoch
                          }
            torch.save(obj=all_states,f=f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

    writer.close()