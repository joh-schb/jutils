# jutils

Some useful utility functions.

Simply install it with

```
pip install git+https://github.com/joh-fischer/jutils.git#egg=jutils
```

## Usage

Please check the `test_all.py` file or the individual python files for usage examples. For example, you
can find functionality for vision in the `jutils/vision` folder, which include depth map colorization,
as well as image and video processing.


## Checkpoints

Pre-trained pytorch checkpoints for the models can be downloaded like this:

```
mkdir checkpoints
cd checkpoints

# SD Autoencoder checkpoint
wget -O sd_ae.ckpt https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=0

# TinyAutoencoderKL checkpoints
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_encoder.pth
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth
```
