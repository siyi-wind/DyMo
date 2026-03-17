"""In this file, we reproduce the MMVAE+ results on the PolyMNIST dataset."""
import torch
from MoPoE_architectures import load_mmnist_classifiers, EncoderImg, DecoderImg

from multivae.data.datasets.mmnist import MMNISTDataset
from pythae.models.base.base_config import BaseAEConfig
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.fids import FIDEvaluator, FIDEvaluatorConfig
from multivae.models.mopoe import MoPoE, MoPoEConfig
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig

# Define paths
DATA_PATH = '/bigdata/siyi/data'
SAVE_PATH = "/home/siyi/project/mm/result/runs"
CKPT_PATH = "/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae"


# Define model
modalities = ["m0", "m1", "m2", "m3", "m4"]
model_config = MoPoEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=512,
    decoders_dist={m: "laplace" for m in modalities},
    decoder_dist_params={m: {"scale": 0.75} for m in modalities},
    beta=2.5,
)


encoders = {
    k: EncoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: DecoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = MoPoE(model_config, encoders=encoders, decoders=decoders)



######## Dataset #########

train_data = MMNISTDataset(data_path=DATA_PATH, split="train")
test_data = MMNISTDataset(data_path=DATA_PATH, split="test")


########## Training #######

trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=1.0e-3,
    steps_predict=1,
    per_device_train_batch_size=256,
    drop_last=True,
    train_dataloader_num_workers=8,
    seed=0,
    output_dir=SAVE_PATH,
)

# Set up callbacks
# wandb_cb = WandbCallback()
# wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae_plus")

# callbacks = [wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    training_config=trainer_config,
    # callbacks=callbacks,
)

if CKPT_PATH is not None:
    print(f'No training. Load model from {CKPT_PATH}')
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(ckpt)
else:
    print("Start training.")
    trainer.train()





#### Validation ####

# Compute Coherences
config = CoherenceEvaluatorConfig(batch_size=512,)
print('=========== Coherence =================')
CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(
        data_path=DATA_PATH + "/clf", device=model.device
    ),
    output=trainer.training_dir,
    eval_config=config,
).eval()
print('\n')

# Compute FID
print('=========== Unconditional FID =================')
config = FIDEvaluatorConfig(
    batch_size=128,
    # wandb_path=wandb_cb.run.path,
    inception_weights_path=DATA_PATH + "/pt_inception-2015-12-05-6726825d.pth",
)

unconditional_fid = FIDEvaluator(
    model, test_data, output=trainer.training_dir, eval_config=config
).eval()
print('\n')

print('=========== Conditional FID =================')
for m in modalities:
    print(f'Generate {m}')
    conditional_fid = FIDEvaluator(
        model, test_data, output=trainer.training_dir, eval_config=config
    ).compute_all_conditional_fids(m)