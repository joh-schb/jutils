import os
import math
import torch
import numpy as np
from sklearn import datasets
from functools import partial
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal


def sample_moons(n, noise=0.05, scale_and_shift=True):
    data, labels = datasets.make_moons(n, noise=noise)
    if scale_and_shift:
        data = data * 3. - 1.
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_circles(n, factor=0.5, noise=0.05):
    data, labels = datasets.make_circles(n, factor=factor, noise=noise)
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_spirals(n, noise=.5):
    n = n // 2      # 2 arms
    ns = np.sqrt(np.random.rand(n,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(ns)*ns + np.random.rand(n,1) * noise
    d1y = np.sin(ns)*ns + np.random.rand(n,1) * noise
    out = np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y))))
    labels = np.hstack((np.zeros(n),np.ones(n)))
    return torch.tensor(out).float(), torch.tensor(labels).long()


def sample_checkerboard(n):
    x1 = np.random.rand(n) * 4 - 2
    x2_ = np.random.rand(n) - np.random.randint(0, 2, n) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    data = np.concatenate([x1[:, None], x2[:, None]], 1)
    labels = (np.floor(x1) + 2 * np.floor(x2)) % 8
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_normal(n):
    data = torch.randn(n, 2)
    return data, torch.zeros(n).long()


def sample_gaussian(n, loc=None, cov=None):
    if loc is None:
        loc = np.array([0, 0])
    if cov is None:
        cov = np.array([[1, 0.8], [0.8, 1]])
    loc = torch.tensor(loc).float()
    cov = torch.tensor(cov).float()
    m = MultivariateNormal(loc, cov)
    data = m.sample((n,))
    return data, torch.zeros(n).long()


def sample_gaussianmixture(n, loc1=(-3, -3), loc2=(3, 3), var1=0.5, var2=1., p1=0.2, p2=0.8):
    multi = torch.multinomial(torch.tensor([p1, p2]).float(), n, replacement=True)
    n1 = (multi == 0).sum()
    n2 = n - n1
    m1 = MultivariateNormal(torch.tensor(loc1).float(), math.sqrt(var1) * torch.eye(2).float())
    m2 = MultivariateNormal(torch.tensor(loc2).float(), math.sqrt(var2) * torch.eye(2).float())
    samples = torch.cat([m1.sample((n1,)), m2.sample((n2,))])
    labels = torch.cat([torch.zeros(n1).long(), torch.ones(n2).long()])
    return samples, labels


def sample_mixture_of_gmm(n, cfg, class_idx: int = None):
    """
    Args:
        n (int): number of samples to generate
        cfg (dict): dictionary with the following keys:
            - class_probs (list): list of class probabilities
            - classes (list): list of class information dictionaries, each
                with the following keys:
                - mode_probs (list): list of mode probabilities
                - params (list): list of dictionaries with the following keys:
                    - loc (list): mean of the mode
                    - cov (list): covariance matrix of the mode
        class_idx (int): if not None, sample only from the specified class
    """
    assert (
        len(cfg["class_probs"]) == len(cfg["classes"])
    ), "Number of class probabilities and classes should be the same!"
    
    samples = []
    labels = []

    # sample class labels
    if class_idx is not None:
        classes_counts = (torch.tensor([class_idx]), torch.tensor([n]))
    else:
        classes_counts = torch.multinomial(
            torch.tensor(cfg["class_probs"]).float(), n, replacement=True
        ).unique(return_counts=True, sorted=True)

    for c_idx, c_count in zip(*classes_counts):
        class_info = cfg["classes"][c_idx.item()]
        
        # sample modes per class
        modes_counts = torch.multinomial(
            torch.tensor(class_info["mode_probs"]).float(), c_count.item(), replacement=True
        ).unique(return_counts=True, sorted=True)

        labels.append(torch.full((c_count.item(),), c_idx))

        for m_idx, m_count in zip(*modes_counts):
            assert (
                len(class_info["mode_probs"]) == len(class_info["params"])
            ), "Number of mode probabilities and modes should be the same!"

            mode_info = class_info["params"][m_idx.item()]
            
            # sample data points per mode
            mode_samples = MultivariateNormal(
                loc=torch.tensor(mode_info["loc"]).float(),
                covariance_matrix=torch.tensor(mode_info["cov"]).float(),
            ).sample((m_count.item(),))

            samples.append(mode_samples)

    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)
    return samples, labels


def mixture_of_gmm_pdf(x, cfg):
    """
    Args:
        x: input data, shape (n_samples, 2)
        cfg: configuration dictionary for the mixture of Gaussians
    """
    assert len(x.shape) == 2 and x.shape[1] == 2, "Input data must be 2D"
    pdf = np.zeros(x.shape[0])
    sum_class_probs = sum(cfg['class_probs'])
    for i, (class_probs, classes) in enumerate(zip(cfg['class_probs'], cfg['classes'])):
        class_prob = class_probs / sum_class_probs
        sum_mode_probs = sum(classes["mode_probs"])
        for mode_probs, params in zip(classes["mode_probs"], classes["params"]):
            mean = params["loc"]
            cov = params["cov"]
            mode_prob = mode_probs / sum_mode_probs
            pdf += class_prob * mode_prob * multivariate_normal(mean, cov).pdf(x)
    return pdf


def sample_2gaussians(n, scale=4., var=0.2, dim=2):
    m = MultivariateNormal(torch.zeros(2), math.sqrt(var) * torch.eye(dim))
    centers = [
        (1, 0),
        (-1, 0),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(2), n, replacement=True)
    data = torch.stack([
        centers[multi[i]] + noise[i]
        for i in range(n)
    ]).float()
    return data, multi.long()


def sample_5gaussians(n, scale=4., var=0.2):
    m = MultivariateNormal(torch.zeros(2), math.sqrt(var) * torch.eye(2))
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (0, 0),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(5), n, replacement=True)
    data = torch.stack([
        centers[multi[i]] + noise[i]
        for i in range(n)
    ]).float()
    return data, multi.long()


def sample_8gaussians(n, scale=5, var=0.1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), math.sqrt(var) * torch.eye(2)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = torch.stack([
        centers[multi[i]] + noise[i]
        for i in range(n)
    ]).float()
    return data, multi.long()


def sample_16gaussians(n, scale=5, var=0.1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), math.sqrt(var) * torch.eye(2)
    )
    centers = []
    for x in range(-2, 2):
        for y in range(-2, 2):
            centers.append((x, y))
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(16), n, replacement=True)
    data = torch.stack([
        centers[multi[i]] + noise[i]
        for i in range(n)
    ]).float()
    return data, multi.long()


def sample_25gaussians(n, scale=8, var=0.1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), math.sqrt(var) * torch.eye(2)
    )
    centers = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            centers.append((x, y))
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(25), n, replacement=True)
    data = torch.stack([
        centers[multi[i]] + noise[i]
        for i in range(n)
    ]).float()
    return data, multi.long()


def sample_n_random_gaussians(n, n_centers=8, noise=0.1, seed=20):
    # sample centers with fixed seed
    cur_state = np.random.get_state()
    np.random.seed(seed)
    centers = np.random.randn(n_centers, 2)
    labels = np.random.randint(n_centers, size=n)
    np.random.set_state(cur_state)
    data = centers[labels] + noise * np.random.randn(n, 2)
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_gaussian_lines(n, n_centers=8, noise=0.1):
    n_per_center = np.ceil(n / n_centers).astype(int)
    t = np.linspace(0, 1, n_centers)
    x = t * n_centers - (n_centers / 2)
    data = np.stack((x, x), axis=1)
    data = np.repeat(data, n_per_center, axis=0)
    data += np.random.randn(*data.shape) * noise
    data = data[:n]
    labels = np.concatenate([np.ones(n_per_center) * i for i in range(n_centers)])[:n]
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_n_concentric_squares(n, n_squares=3, noise=0.05, scale=0.5):
    data = []
    labels = []
    samples_per_sq = _divide_samples(n, n_squares)
    for i, sq_samples in enumerate(samples_per_sq):
        sq_samples = np.ceil(sq_samples / 4).astype(int)  # we need 4 sides per square
        square_data = np.zeros((4 * sq_samples, 2))
        t = np.linspace(-1, 1, sq_samples)
        square_data[:sq_samples] = np.c_[t, -np.ones_like(t)]  # bottom side
        square_data[sq_samples:2*sq_samples] = np.c_[np.ones_like(t), t]  # right side
        square_data[2*sq_samples:3*sq_samples] = np.c_[t[::-1], np.ones_like(t)]  # top side
        square_data[3*sq_samples:] = np.c_[np.ones_like(t)*-1, t[::-1]]  # left side
        square_data = square_data * (scale * (i + 1))  # scale the data to adjust the size of each square
        data.append(square_data)
        labels.append(np.ones(square_data.shape[0]) * i)
    data = np.vstack(data)[:n]
    data += np.random.normal(scale=noise, size=data.shape)  # add noise
    data *= 2
    labels = np.concatenate(labels)[:n]
    return torch.tensor(data).float(), torch.tensor(labels).long()


def sample_n_concentric_circles(n, n_circles=3, noise=0.1, scale=0.5):
    data = []
    labels = []
    samples_per_circle = _divide_samples(n, n_circles)
    for i, sq_samples in enumerate(samples_per_circle):
        circle_data, _ = datasets.make_circles(
            n_samples=sq_samples, factor=.99, noise=noise / (i + 1)
        )
        circle_data = circle_data * (i + scale)
        data.append(circle_data)
        circle_labels = np.ones(sq_samples) * i
        labels.append(circle_labels)
    data = np.vstack(data)[:n]
    labels = np.hstack(labels)[:n]
    return torch.tensor(data).float(), torch.tensor(labels).long()


def create_square_data(n=1000, scale=3.0):  # 3 is set by the spread of the gaussian and spiral
    """Create points uniformly distributed in a square"""
    # Generate uniform points in a square
    points = (torch.rand(n, 2) * 2 - 1) * scale
    return points


def sample_heart(n=1000, scale=3.0):
    """ Heart shape distribution, adapted from https://drscotthawley.github.io/blog/posts/FlowModels.html """
    square_points = create_square_data(n, scale=1.0)

    # Calculate the heart-shaped condition for each point
    x, y = square_points[:, 0], square_points[:, 1]
    heart_condition = x**2 + ((5 * (y + 0.25) / 4) - torch.sqrt(torch.abs(x)))**2 <= 1

    # Filter out points that don't satisfy the heart-shaped condition
    heart_points = square_points[heart_condition]

    # If we don't have enough points, generate more
    while len(heart_points) < n:
        new_points = create_square_data(n - len(heart_points), scale=1)
        x, y = new_points[:, 0], new_points[:, 1]
        new_heart_condition = x**2 + ((5 * (y + 0.25) / 4) - torch.sqrt(torch.abs(x)))**2 <= 1
        new_heart_points = new_points[new_heart_condition]
        heart_points = torch.cat([heart_points, new_heart_points], dim=0)

    heart_points *= scale
    return heart_points[:n], torch.zeros(n).long()


def sample_smiley(n=1000, scale=2.5):
    """ Smiley face, adapted from https://drscotthawley.github.io/blog/posts/FlowModels.html """
    points = []
    labels = []
    
    n_part = n//3 + 1

    # Face circle
    # angles = 2 * np.pi * torch.rand(n_points//2+20)
    # r = scale + (scale/10)*torch.sqrt(torch.rand(n_points//2+20))
    # points.append(torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1))

    # Eyes (small circles at fixed positions)
    eye_left = torch.randn(n_part, 2) * 0.2 + torch.tensor([-1, 0.9]) * scale * 0.4
    points.append(eye_left)
    labels.append(torch.ones(n_part).long() * 0)

    eye_right = torch.randn(n_part, 2) * 0.2 + torch.tensor([1, 0.9]) * scale * 0.4
    points.append(eye_right)
    labels.append(torch.ones(n_part).long() * 1)

    # Smile (arc in polar coordinates)
    theta = -np.pi/6 - 2*np.pi/3*torch.rand(n_part)
    r_smile = scale * 0.6 + (scale/4)* torch.rand_like(theta)
    points.append(torch.stack([r_smile * torch.cos(theta), r_smile * torch.sin(theta)], dim=1))
    labels.append(torch.ones(n_part).long() * 2)

    points = torch.cat(points, dim=0)  # concatenate first
    labels = torch.cat(labels, dim=0)  # concatenate first
    return points[:n,:], labels[:n]


def _divide_samples(n_samples, n_groups):
    group_sizes = [i for i in range(1, n_groups + 1)]
    group_sizes = [int(size / sum(group_sizes) * n_samples) for size in group_sizes]
    group_sizes[-1] = n_samples - sum(group_sizes[:-1])
    return group_sizes


# dictionary of all available 2D datasets
DATASETS = {
    "moons": sample_moons,
    "circles": sample_circles,
    "spirals": sample_spirals,
    "checkerboard": sample_checkerboard,
    "normal": sample_normal,
    "gaussian": sample_gaussian,
    "gaussianmixture": sample_gaussianmixture,
    "5gaussians": sample_5gaussians,
    "8gaussians": sample_8gaussians,
    "16gaussians": sample_16gaussians,
    "25gaussians": sample_25gaussians,
    "randomgaussians": sample_n_random_gaussians,
    "gaussianlines": sample_gaussian_lines,
    "consquares": sample_n_concentric_squares,
    "concircles": sample_n_concentric_circles,
    "heart": sample_heart,
    "smiley": sample_smiley,
    "gmm_cfg": sample_mixture_of_gmm,
}


class Dataset2D:
    def __init__(self, dataset, transform=None, verbose=False, **kwargs):
        """
        Args:
            dataset: str, name of the dataset to sample (print available datasets with `DATASETS.keys()`)
            transform: callable, function to apply to the sampled data
        """
        self.dataset = dataset
        self.transform = transform or (lambda x: x)
        if dataset not in DATASETS.keys():
            raise ValueError(f"Unknown dataset: {dataset}")
        self.sample_fn = partial(DATASETS[dataset], **kwargs)

        samples_, labels_ = self.sample(50_000, labels=True)
        self.mean = samples_.mean(0)
        self.std = samples_.std(0)
        self.n_classes = len(torch.unique(labels_))
        
        if verbose:
            print(f"Dataset2D: '{dataset}' with kwargs: {kwargs}")

    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        return x * self.std + self.mean
    
    def sample(self, n, shuffle=True, normalize=False, labels=False, **sample_kwargs):
        x, y = self.sample_fn(n, **sample_kwargs)
        x = self.transform(x)
        if normalize:
            x = self.normalize(x)
        if shuffle:
            perm = torch.randperm(n)
            x = x[perm]
            y = y[perm]
        if labels:
            return x, y
        return x


if __name__ == "__main__":
    torch.manual_seed(2024)
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt

    currentdir = os.path.dirname(__file__)
    main_dir = os.path.dirname(os.path.dirname(currentdir))

    # ===== sample and plot 2D datasets =====
    n_datasets = len(DATASETS)
    N_SAMPLES = 10_000
    print(f"Number of datasets: {n_datasets}")

    ncols = 3
    nrows = np.ceil(n_datasets / ncols).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))

    for i, ds_name in enumerate(DATASETS.keys()):
        
        kwargs = {}
        if ds_name == "gmm_cfg":
            cfg = OmegaConf.load(os.path.join(currentdir, "gmm_configs.yaml"))
            kwargs.update({"cfg": cfg.test_gmm})
        
        dataset = Dataset2D(ds_name, **kwargs)
        
        samples, labels = dataset.sample(N_SAMPLES, labels=True)
        ax = axes.flatten()[i]
        ax.scatter(samples[:, 0], samples[:, 1], s=5, c=labels, cmap="jet")
        ax.set_title(ds_name)
        ax.axis('equal')

    # ===== Save Figure =====
    plt.savefig(os.path.join(main_dir, "assets", "datasets_2d.jpg"), bbox_inches="tight")
