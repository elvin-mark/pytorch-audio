import torch
import torchaudio
import os
from scipy.io import wavfile

EXTRA_INFO_SPOKENDIGITS = {"labels": [str(i) for i in range(
    10)], "orig_freq": 8000, "new_freq": 8000, "audio_length": 16000, "url": "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.9.tar.gz", "train_size": 2000, "test_size": 500}
EXTRA_INFO_GTZAN = {"labels": ["blues", "classical", "country", "disco",
                               "hiphop", "jazz", "metal", "pop", "reggae", "rock"], "orig_freq": 22050, "new_freq": 8000, "audio_length": 240000}

EXTRA_INFO_SPEECHCOMMANDS = {"labels": ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
                                        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'], "orig_freq": 16000, "new_freq": 8000, "audio_length": 16000}

ROOT_DIR = "./data/"


def download_and_extract(url):
    fn = os.path.join(ROOT_DIR, "spokendigits.tar.gz")
    fpath = os.path.join(ROOT_DIR, "free-spoken-digit-dataset-1.0.9")
    print(fn, fpath)
    if os.path.exists(fn):
        print("file already downloaded!")
    else:
        os.system(f"wget {url} -O {fn}")
    if os.path.exists(fpath):
        print("file verified!")
    else:
        os.system(f"tar xzf {fn} --directory={ROOT_DIR}")
    return fn, fpath


def spokendigits_dataloader(args):
    fn, fpath = download_and_extract(EXTRA_INFO_SPOKENDIGITS["url"])

    class SpokenDigitsDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.root = os.path.join(fpath, "recordings")
            self.list_elems = os.listdir(self.root)

        def __getitem__(self, idx):
            audio_fn = self.list_elems[idx]
            audio_fpath = os.path.join(self.root, audio_fn)
            rate, data = wavfile.read(audio_fpath)
            label, speaker, _ = audio_fn.split("_")
            return torch.from_numpy(data/4000).reshape(1, -1).float(), rate, label

        def __len__(self):
            return len(self.list_elems)

    ds = SpokenDigitsDataset()
    train_ds, test_ds = torch.utils.data.random_split(
        ds, [EXTRA_INFO_SPOKENDIGITS["train_size"], EXTRA_INFO_SPOKENDIGITS["test_size"]])
    label_dict = {k: v for v, k in enumerate(
        EXTRA_INFO_SPOKENDIGITS["labels"])}
    resampling = torchaudio.transforms.Resample(
        orig_freq=EXTRA_INFO_SPOKENDIGITS["orig_freq"], new_freq=EXTRA_INFO_SPOKENDIGITS["new_freq"])

    def collate_fn(batch):
        wavdata = []
        labels = []
        for wavdata_, _, labels_ in batch:
            wavdata_ = resampling(wavdata_)
            _, N = wavdata_.shape
            if N < EXTRA_INFO_SPOKENDIGITS["audio_length"]:
                wavdata_ = torch.functional.F.pad(
                    wavdata_, (0, EXTRA_INFO_SPOKENDIGITS["audio_length"] - N), mode="constant", value=0.0)
            wavdata.append(
                wavdata_[:, :EXTRA_INFO_SPOKENDIGITS["audio_length"]])
            labels.append(label_dict[labels_])
        return torch.stack(wavdata), torch.tensor(labels).long()

    raw_test_ds = test_ds

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_SPOKENDIGITS


def gtzan_dataloader(args):
    label_dict = {k: v for v, k in enumerate(EXTRA_INFO_GTZAN["labels"])}
    resampling = torchaudio.transforms.Resample(
        orig_freq=EXTRA_INFO_GTZAN["orig_freq"], new_freq=EXTRA_INFO_GTZAN["new_freq"])

    def collate_fn(batch):
        wavdata = []
        labels = []
        for wavdata_, _, labels_ in batch:
            wavdata_ = resampling(wavdata_)
            _, N = wavdata_.shape
            if N < EXTRA_INFO_GTZAN["audio_length"]:
                wavdata_ = torch.functional.F.pad(
                    wavdata_, (0, EXTRA_INFO_GTZAN["audio_length"] - N), mode="constant", value=0.0)
            wavdata.append(wavdata_[:, :EXTRA_INFO_GTZAN["audio_length"]])
            labels.append(label_dict[labels_])
        return torch.stack(wavdata), torch.tensor(labels).long()

    train_ds = torchaudio.datasets.GTZAN(
        ROOT_DIR, download=True, subset="training")
    test_ds = torchaudio.datasets.GTZAN(
        ROOT_DIR, download=True, subset="testing")
    raw_test_ds = test_ds

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_GTZAN


def speechcommands_dataloader(args):
    label_dict = {k: v for v, k in enumerate(
        EXTRA_INFO_SPEECHCOMMANDS["labels"])}
    resampling = torchaudio.transforms.Resample(
        orig_freq=EXTRA_INFO_SPEECHCOMMANDS["orig_freq"], new_freq=EXTRA_INFO_SPEECHCOMMANDS["new_freq"])

    def collate_fn(batch):
        wavdata = []
        labels = []
        for wavdata_, _, labels_, _, _ in batch:
            wavdata_ = resampling(wavdata_)
            _, N = wavdata_.shape
            if N < EXTRA_INFO_SPEECHCOMMANDS["audio_length"]:
                wavdata_ = torch.functional.F.pad(
                    wavdata_, (0, EXTRA_INFO_SPEECHCOMMANDS["audio_length"] - N), mode="constant", value=0.0)
            wavdata.append(
                wavdata_[:, :EXTRA_INFO_SPEECHCOMMANDS["audio_length"]])
            labels.append(label_dict[labels_])
        return torch.stack(wavdata), torch.tensor(labels).long()

    train_ds = torchaudio.datasets.SPEECHCOMMANDS(
        ROOT_DIR, download=True, subset="training")
    test_ds = torchaudio.datasets.SPEECHCOMMANDS(
        ROOT_DIR, download=True, subset="testing")
    raw_test_ds = test_ds

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_SPEECHCOMMANDS


def create_dataloader(args):
    if args.dataset == "gtzan":
        return gtzan_dataloader(args)
    elif args.dataset == "speechcommands":
        return speechcommands_dataloader(args)
    elif args.dataset == "spokendigits":
        return spokendigits_dataloader(args)
    else:
        return None
