import torch
import torch.nn as nn
import pickle

from cli_parser import create_train_parser
from models import create_model
from datasets import create_dataloader
from optim import create_optim
from scheduler import create_lr_scheduler
from utils import train, WebLogger, test_audio, get_model_graph, draw_graph, \
    get_graph_structure

import os

args = create_train_parser()
if not os.path.exists("trained_models"):
    os.mkdir("trained_models")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if not os.path.exists("labels"):
    os.mkdir("labels")

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

if args.customize:
    from customize import create_model_customize
    model = create_model_customize(args).to(dev)
else:
    model = create_model(args).to(dev)

if args.start_model is not None:
    try:
        model.load_state_dict(torch.load(args.start_model, map_location=dev))
    except:
        print("Could not load specified model. Using random model")

if args.customize:
    from customize import create_dataloader_customize
    train_dl, test_dl, test_ds, raw_test_ds, extra_info = create_dataloader_customize(
        args)
else:
    train_dl, test_dl, test_ds, raw_test_ds, extra_info = create_dataloader(
        args)

optim = create_optim(args, model)
lr_sched = create_lr_scheduler(args, optim)
crit = nn.CrossEntropyLoss()

web_logger = None
if args.dashboard:
    print("Preparing Dashboard Logger")
    web_logger = WebLogger(args.port, customize_url=args.url)

print("Start Training ...")
hist = train(model, train_dl, test_dl, crit, optim,
             args.epochs, dev, lr_sched=lr_sched, logging=args.logging, csv=args.csv, dashboard=args.dashboard, web_logger=web_logger, checkpoint=args.checkpoint)

if args.csv and hist is not None:
    print("Saving CSV Record")
    hist.to_csv(f"{args.model}_record.csv")

if args.save_model:
    print("Saving model ...")
    torch.save(model.state_dict(), f"trained_models/{args.model}.ckpt")

if args.samples:
    print("Generating Sample Audio Test")
    results = test_audio(
        model, test_dl, extra_info["labels"], extra_info["new_freq"], dev)
    if web_logger is not None:
        web_logger.send_samples(results)

if args.labels:
    with open("labels/dataset_labels.pkl", "wb") as f:
        pickle.dump(extra_info["labels"], f)

if web_logger is not None:
    try:
        nodes_dict, layers_dict = get_model_graph(
            model, torch.zeros((1, 1, args.audio_length)), dev)
        graph_image = draw_graph(nodes_dict, layers_dict)
        graph_struct = get_graph_structure(nodes_dict, layers_dict)
        results = {"graph_img": graph_image, "graph_struct": graph_struct}
        web_logger.send_model(results)
    except:
        print("Could not generate the model graph!")
