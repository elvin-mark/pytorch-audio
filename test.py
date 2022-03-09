import torch
import torch.nn as nn

from cli_parser import create_test_parser
from models import create_model
from datasets import create_dataloader
from utils import evaluate, WebLogger, test_audio, draw_graph, get_model_graph, \
    get_graph_structure
import os
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io

args = create_test_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for testing")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for testing. It can be a little bit slow")

if args.customize:
    from customize import create_model_customize
    model = create_model_customize(args).to(dev)
else:
    model = create_model(args).to(dev)

model.load_state_dict(torch.load(args.model_path, map_location=dev))

if args.customize:
    from customize import create_dataloader_customize
    train_dl, test_dl, test_ds, raw_test_ds, extra_info = create_dataloader_customize(
        args)
else:
    train_dl, test_dl, test_ds, raw_test_ds, extra_info = create_dataloader(
        args)

crit = nn.CrossEntropyLoss()

print("Start Testing ...")
train_loss, train_acc = evaluate(model, train_dl, crit, dev)
test_loss, test_acc = evaluate(model, test_dl, crit, dev)

print(
    f"train_loss: {train_loss}, train_acc: {train_acc:.2f}%, test_loss: {test_loss}, test_acc: {test_acc:.2f}%")

if args.dashboard:
    print("Preparing Dashboard Logger")
    web_logger = WebLogger(args.port, customize_url=args.url)
    data = {"epoch": 0, "lr": 0, "train_loss": train_loss,
            "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc}
    web_logger.send_data(data)

if args.samples:
    print("Generating Sample Images Test")
    results = test_audio(
        model, test_ds, raw_test_ds, extra_info["labels"], dev)
    if web_logger is not None:
        web_logger.send_samples(results)

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
