# latent-bridge

# Recuperare dal checkpoint 
```bash
python scripts/train.py \
    model=pixel_mlp \
    trainer.resume_from_checkpoint=checkpoints/PixelMLP/model_epoch_
```

```bash
python scripts/train.py \
    model=spatial_cnn \
    trainer.resume_from_checkpoint=checkpoints/SpatialCNN/model_epoch_
```