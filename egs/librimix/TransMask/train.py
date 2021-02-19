import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from asteroid import TransMask
from asteroid import DPTrans
# from asteroid.engine import schedulers

# from asteroid.data.wham_dataset import WhamDataset
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--config", default="local/conf.yml", help="config yaml file")


def main(conf):
    # train_set = WhamDataset(
    #     conf["data"]["train_dir"],
    #     conf["data"]["task"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     segment=conf["data"]["segment"],
    #     nondefault_nsrc=conf["data"]["nondefault_nsrc"],
    # )
    # val_set = WhamDataset(
    #     conf["data"]["valid_dir"],
    #     conf["data"]["task"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     nondefault_nsrc=conf["data"]["nondefault_nsrc"],
    # )
    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["masknet"]["n_src"],
        segment=conf["data"]["segment"],
    )
    print(conf["data"]["train_dir"])

    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["masknet"]["n_src"],
        segment=conf["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    # Update number of source values (It depends on the task)
    conf["masknet"].update({"n_src": train_set.n_src})

    # TODO params
    # model = TransMask(**conf["filterbank"], **conf["masknet"])
    model = DPTrans(**conf["filterbank"], **conf["masknet"], sample_rate=conf['data']['sample_rate'])

    # from torchsummary import summary
    # model.cuda()
    # summary(model, (24000,))
    # import pdb
    # pdb.set_trace()

    optimizer = make_optimizer(model.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    # # TODO warmup for transformer
    # from asteroid.engine.schedulers import DPTNetScheduler
    # schedulers = {
    #     "scheduler": DPTNetScheduler(
    #         # optimizer, len(train_loader) // conf["training"]["batch_size"], 64
    #         # optimizer, len(train_loader), 64,
    #         optimizer, len(train_loader), 128,
    #         stride=2,
    #         # exp_max=0.0004 * 16,
    #         # warmup_steps=1000
    #     ),
    #     "interval": "batch",
    # }

    # from torch.optim.lr_scheduler import ReduceLROnPlateau
    # if conf["training"]["half_lr"]:
    #     print('Use ReduceLROnPlateau halflr...........')
    #     schedulers = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    # checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    if conf["training"]["cont"]:
        from glob import glob
        ckpts = glob('%s/*.ckpt' % checkpoint_dir)
        ckpts.sort()
        latest_ckpt = ckpts[-1]
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            gpus=gpus,
            distributed_backend=distributed_backend,
            limit_train_batches=1.0,  # Useful for fast experiment
            gradient_clip_val=conf["training"]["gradient_clipping"],
            resume_from_checkpoint=latest_ckpt
        )
    else:
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            gpus=gpus,
            distributed_backend=distributed_backend,
            limit_train_batches=1.0,  # Useful for fast experiment
            gradient_clip_val=conf["training"]["gradient_clipping"],
        )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    # state_dict = torch.load('exp/train_transmask_rnn_acous_gelu_6layer_peconv_stride2_batch6/_ckpt_epoch_208.ckpt')
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open(parser.parse_args().config) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
