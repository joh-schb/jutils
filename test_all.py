import os


def test_image(show_output=True):
    from jutils import get_original_reconstruction_image, alpha_compose
    from PIL import Image
    import numpy as np

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    im_fp = os.path.join(cur_dir, 'assets', 'image.jpg')

    # alpha_compose(bg_im, fg_im, alpha=0.5)
    im_a = np.array(Image.open(im_fp))
    im_b = (np.random.randn(*im_a.shape) * 127.5 + 127.5).astype(np.uint8)
    im_c = alpha_compose(im_a, im_b, alpha=0.4)
    if show_output:
        im_c.show()

    # get_original_reconstruction_image(x, x_hat, channels_first=True)
    im_orig = np.array(Image.open(im_fp))
    im_orig = np.stack((im_orig, im_orig), axis=0)              # batched images
    im_rec = (np.random.randn(*im_orig.shape) * 127.5 + 127.5).astype(np.uint8)
    im_orig_rec = get_original_reconstruction_image(im_orig, im_rec)
    print("Shape of original vs reconstructed image:", im_orig_rec.shape)
    if show_output:
        Image.fromarray(im_orig_rec).show()


def test_video():
    from jutils import animate_video, save_as_gif
    import numpy as np

    vid = np.random.randint(0, 255, (10, 256, 256, 3))

    # display video in jupyter notebook
    animate_video(vid)

    # save video as gif
    save_as_gif(vid, 'logs/test.gif')


def test_helpers():
    from jutils import convert_size, Namespace, instantiate_from_config

    # convert_size(size_bytes)
    print("convert_size(1_000_000_000):", convert_size(1_000_000_000))

    # Namespace()
    self = Namespace(a=1)
    self.b = 3
    print("Namespace:", self.a, self.b)

    # instantiate_from_config(config)
    cfg = dict(target="jutils.helpers.Namespace", params=dict(a=1, b=2))
    print("instantiate_from_config(cfg):", instantiate_from_config(cfg))

    # suppress_stdout()
    from jutils import suppress_stdout
    with suppress_stdout():
        print("This will not be printed!")
    print("This will be printed!")


def test_log():
    from jutils import get_logger
    my_logger = get_logger(log_to_file=True, log_to_stdout=True)
    my_logger.info("Hello world!")
    my_logger.warning("This is a warning!")


def test_timing():
    import time
    from jutils import timer, Timer, get_time

    # timer(start, end)
    t0 = time.time()
    time.sleep(0.2)
    print("timer(start, end):", timer(t0, time.time()))

    # Timer
    with Timer() as t:
        time.sleep(0.2)
    print("Timer():", t.time)

    # timing
    print("get_time():", get_time())


def test_torchy():
    import torch
    from jutils import get_tensor_size, count_parameters

    # get_tensor_size(tensor)
    x = torch.randn((3, 224, 224))
    print("get_tensor_size(x):", get_tensor_size(x))

    # count_parameters(model)
    my_model = torch.nn.Linear(10, 10)
    print("count_parameters(model):", count_parameters(my_model))


def test_transformer():
    import torch
    from jutils.nn import TransformerLayer
    transformer = TransformerLayer(768, d_cond_norm=128, d_cross=64)
    kwargs = dict(
        x=torch.randn((1, 256, 768)),
        pos=torch.randn((1, 256, 2)),
        cond_norm=torch.randn((1, 1, 128)),
        x_cross=torch.randn((1, 256, 64)),
    )
    out = transformer(**kwargs)
    print(f"TransformerLayer: in.shape={kwargs['x'].shape} - out.shape={out.shape}")


def test_rope():
    import torch
    from jutils.nn.rope import make_axial_pos_1d, make_axial_pos_2d, make_axial_pos_3d

    f, h, w = 10, 4, 5
    pos_1d = make_axial_pos_1d(f)
    print(f"make_axial_pos_1d({f}):", pos_1d.shape)
    pos_2d = make_axial_pos_2d(h, w)
    print(f"make_axial_pos_2d({h}, {w}):", pos_2d.shape)
    pos_3d = make_axial_pos_3d(f, h, w)
    print(f"make_axial_pos_3d({f}, {h}, {w}):", pos_3d.shape)


def test_all_files():
    import glob
    import subprocess
    root_dir = os.getcwd()

    # find all python files in jutils folder
    python_files = glob.glob(os.path.join(root_dir, 'jutils', '**', '*.py'), recursive=True)

    # skip some files
    skip_files = ['min_DDP.py']
    python_files = [pf for pf in python_files if os.path.basename(pf) not in skip_files]

    # execute each python file
    for pf in python_files:
        if '__init__.py' in pf:
            continue
        print(f"[****] Executing: {pf}")
        subprocess.run(['python', pf])


if __name__ == "__main__":
    # test_image(show_output=False)
    # test_video()
    # test_helpers()
    # test_log()
    # test_timing()
    # test_torchy()
    # test_transformer()
    # test_rope()
    test_all_files()
