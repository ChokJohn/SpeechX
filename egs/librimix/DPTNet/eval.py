import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
<<<<<<< HEAD

from asteroid import DPRNNTasNet
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
=======
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import DPTNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea


parser = argparse.ArgumentParser()
parser.add_argument(
<<<<<<< HEAD
=======
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
<<<<<<< HEAD
    "--test_dir", type=str, required=True, help="Test directory including the json files"
=======
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results" " will be stored",
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
<<<<<<< HEAD
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = DPRNNTasNet.from_pretrained(model_path)
=======
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
ASR_MODEL_PATH = (
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)


def update_compute_metrics(compute_wer, metric_list):
    if not compute_wer:
        return metric_list
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ModuleNotFoundError:
        import warnings

        warnings.warn("Couldn't find espnet installation. Continuing without.")
        return metric_list
    return metric_list + ["wer"]


def main(conf):
    compute_metrics = update_compute_metrics(conf["compute_wer"], COMPUTE_METRICS)
    anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent / "test_annotations.csv")
    wer_tracker = (
        MockWERTracker() if not conf["compute_wer"] else WERTracker(ASR_MODEL_PATH, anno_df)
    )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = DPTNet.from_pretrained(model_path)
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
<<<<<<< HEAD
    test_set = WhamDataset(
        conf["test_dir"],
        conf["task"],
        sample_rate=conf["sample_rate"],
        nondefault_nsrc=model.masker.n_src,
        segment=None,
=======
    test_set = LibriMix(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
<<<<<<< HEAD
    ex_save_dir = os.path.join(conf["exp_dir"], "examples/")
=======
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
<<<<<<< HEAD
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix[None, None])
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix[None].cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
=======
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(mix.unsqueeze(0))
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
<<<<<<< HEAD
            metrics_list=compute_metrics,
        )
        utt_metrics["mix_path"] = test_set.mix[idx][0]
=======
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np,
                clean=sources_np,
                estimate=est_sources_np_normalized,
                wav_id=ids,
                sample_rate=conf["sample_rate"],
            )
        )
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
<<<<<<< HEAD
            sf.write(local_save_dir + "mixture.wav", mix_np[0], conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx + 1), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx + 1),
=======
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np_normalized):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
<<<<<<< HEAD
    all_metrics_df.to_csv(os.path.join(conf["exp_dir"], "all_metrics.csv"))
=======
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
<<<<<<< HEAD
    print("Overall metrics :")
    pprint(final_results)
    with open(os.path.join(conf["exp_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)
    model_dict = torch.load(model_path, map_location="cpu")

=======

    print("Overall metrics :")
    pprint(final_results)
    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report_as_markdown()
        print(wer_card)
        # Save the report
        with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
            f.write(wer_card)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
<<<<<<< HEAD

=======
>>>>>>> 210b5e4eb8ce24fe25780e008c89a4bb71bbd0ea
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
