# pytorch-audio

Simple trainer cli for speech recognition models.

## How to use

### Training

Run the server script in case you want to use the dashboard. This will create a local server using flask.

```
python server.py
```

In the dashboard you can see the evolution of the training of your network model, predictions on some samples audios and an sketch of your model!.
![dashboard](samples/dashboard.png?raw=true "Dashboard")
![samples](samples/samples.png?raw=true "Samples")
![modelgraph](samples/modelgraph.png?raw=true "Model Graph")

You can also use the following command line using ngrok if you want your dashboard to be available outside your local network. (If you want to use it from google colab for example)

```
ngrok http PORT
```

Run this line to start training

```
python train.py \
  --dataset {gtzan,speechcommands,spokendigits} \
  --audio-length AUDIO_LENGTH \
  --root ROOT \
  --num-classes NUM_CLASSES \
  --model {simple_general_cnn,simple_general_resnet} \
  --batch-size BATCH_SIZE \
  --gpu \
  --optim OPTIM \
  --lr LR \
  --epochs EPOCHS \
  --logging \
  --no-logging \
  --save-model \
  --csv \
  --dashboard \
  --port PORT \
  --checkpoint CHECKPOINT \
  --samples \
  --url URL \
  --start-model START_MODEL \
  --save-labels \
  --customize \
  --sched {none,step} \
  --step-size STEP_SIZE \
  --gamma GAMMA \
  --split-dataset \
  --train-size TRAIN_SIZE
```

### Audio Folder

Using an audio folder containing all **wav** files in the following 2 formats.

- Using split-dataset flag:

  By using the split-dataset flag, the command itself will manage to separate the given dataset into training samples and testing samples.

```
root/
  class1/
    audio1.wav ...
  class2/
    audio1.wav ...
  ...
```

- Without using split-dataset flag:

  This requires the user to manually had divided the sample images into 2 different folders: one for training and another for testing as in the following structure.

```
root/
  train/
    class1/
      audio1.wav ...
    class2/
      audio1.wav ...
  test/
    class1/
      audio1.wav ...
    class2/
      audio1.wav ...
```

Example on how to train

```
python train.py --dataset audio_folder --root PATH_TO_ROOT --model simple_general_cnn --num-classes NUMBER_OF_CLASSES --save-model --epochs 10 --audio-length 16000
```

### Customize models and dataloaders

We can use a customize model and dataloader. Create a customize.py file (using the following template) in the root folder of this repo. (It is important for the functions to have these names as it will be imported internally by the script)

```python
def create_model_customize(args):
  # ...
  # your code
  return model

def create_dataloader_customize(args):
  # ...
  # your code
  # train_dl: Train dataloader
  # test_dl: Test dataloader
  # test_ds: Test dataset
  # raw_test_ds: Raw test dataset (without any transformation, used for the samples)
  # extra_info: dictonary of extra information ({ "labels": ["LABEL1","LABEL2",...],...})
  return train_dl, test_dl, test_ds, raw_test_ds, extra_info
```

Example on how to train

```
python train.py --customize --epochs 10 --save-model
```

For more information about this script

```
python train.py -h
```

You can find more information about how to train using this script in the [tutorial folder](https://github.com/elvin-mark/pytorch_trainer/tree/main/tutorial).

### Testing

Run this line to test a trained model

```
python test.py \
  --dataset {gtzan,speechcommands,spokendigits} \
  --audio-length AUDIO_LENGTH \
  --root ROOT \
  --num-classes NUM_CLASSES \
  --model {simple_general_cnn,simple_general_resnet} \
  --batch-size BATCH_SIZE \
  --gpu \
  --model-path MODEL_PATH \
  --samples \
  --url URL \
  --dashboard \
  --port PORT \
  --customize \
  --split-dataset \
  --train-size TRAIN_SIZE

```

Run this line for more information

```
python test.py -h
```
