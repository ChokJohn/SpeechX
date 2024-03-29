import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet, TransMask
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from time import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--file_path", default="", help="sythesize single file")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)

# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]
compute_metrics = ['sdr']


def inference_wav(file_path, conf, model_device, model, ex_save_dir):
    wavid = os.path.basename(file_path).split('.')[0]
    mixture, _ = sf.read(file_path, dtype="float32")
    mixture = torch.from_numpy(mixture)
    mix = tensors_to_device(mixture, device=model_device)
    mul = 1
    mix = mix.view(-1, 1).repeat(1, mul).view(-1)
    mix_np = mix.cpu().data.numpy()
    est_sources = model(mix.unsqueeze(0))
    est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
    local_save_dir = os.path.join(ex_save_dir, "ex/")
    os.makedirs(local_save_dir, exist_ok=True)
    print(local_save_dir)
    for src_idx, est_src in enumerate(est_sources_np):
        est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
        sf.write(
            local_save_dir + "{}_s{}_estimate.wav".format(wavid, src_idx),
            est_src,
            conf["sample_rate"],
        )
    return


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = TransMask.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    if conf['file_path'] == '':
        test_set = LibriMix(
            csv_dir=conf["test_dir"],
            task=conf["task"],
            sample_rate=conf["sample_rate"],
            n_src=conf["train_conf"]["masknet"]["n_src"],
            segment=None,
        )  # Uses all segment length
        # Used to reorder sources only
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1 and conf['file_path']=='':
        conf["n_save_ex"] = len(test_set)
        save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    else:
        save_idx = 0
    series_list = []
    torch.no_grad().__enter__()
    sdr=0
    rtf=0
    if conf['file_path'] != '':
        file_path=conf['file_path']
        if os.path.isdir(file_path):
            wavs=[os.path.join(file_path, wav) for wav in os.listdir(file_path) if '.wav' in wav]
            for wav in wavs:
                inference_wav(wav, conf, model_device, model, ex_save_dir)
        else:
            inference_wav(file_path, conf, model_device, model, ex_save_dir)
        return

    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)

        mul=8
        mix=mix.view(-1,1).repeat(1,mul).view(-1)
        sources=sources.repeat(1, mul)

        #print('DEVICE')
        #print(model_device)
        ss=time()
        est_sources = model(mix.unsqueeze(0))
        dur=time()-ss
        ll=len(mix)/8000
        rtf+=(dur/ll)
        print(rtf/(idx+1))
        #import pdb;pdb.set_trace()

        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )

        sdr+=utt_metrics['sdr']
        print(sdr/(idx+1))

        utt_metrics["mix_path"] = test_set.mixture_path
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    # publishable = save_publishable(
    save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
