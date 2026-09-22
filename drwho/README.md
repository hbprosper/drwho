# `drwho` — package contents

Shared utilities used by several repositories under
[github.com/hbprosper](https://github.com/hbprosper). Nothing here is specific to
any one project; anything that *is* project-specific belongs in that project's
own package.

Install it indirectly, by declaring it in a dependent repository's
`pyproject.toml`:

```toml
dependencies = [
    "drwho @ git+https://github.com/hbprosper/drwho.git@main",
]
```

or directly:

```bash
pip install "drwho @ git+https://github.com/hbprosper/drwho.git@main"
```

See `docs/drwho-scheme.pdf` for the full scheme and the release protocol.

---

## Layout

| path | what it holds |
|---|---|
| `__init__.py` | version number and lazy submodule loading |
| `nn.py` | neural-network models, losses, training objective, run configuration |
| `data.py` | datasets, a fast data loader, sampling, dataset download |
| `monitor.py` | real-time monitoring of loss curves during training |
| `kdtree.py` | KD-tree via recursive binary partitioning |
| `bin/monitor_losses.py` | the `monlosses` command-line loss viewer |

Submodules are imported **lazily** (PEP 562), so `import drwho` is cheap and does
not pull in `torch`:

```python
import drwho
print(drwho.__version__)

cfg = drwho.nn.Config('myrun')     # torch is imported at this point
```

Explicit imports work as usual:

```python
from drwho import nn as mlp
from drwho import data as dat
from drwho import monitor as mon
```

---

## `nn.py` — models, losses, configuration

Built on PyTorch. Written for the Machine Learning in Physics course at Florida
State University and for the AIMS PINN black-hole project (with Claire David and
Tlotlo Oepeng).

### Models

| object | signature | notes |
|---|---|---|
| `Model` | `Model()` | Base class for the models below. Adds `save(paramsfile)` / `load(paramsfile)` on top of `nn.Module`. |
| `FCNN` | `FCNN(n_inputs, n_hidden=4, n_width=32, f_hidden=Sin, f_output=None)` | Fully-connected network. |
| `ResNet` | `ResNet(n_input, n_width, f_hidden=Sin())` | Residual network. |
| `ELM` | `ELM(n_inputs, n_width, n_outputs, nonlinearity=Sin)` | Extreme learning machine. Fitted in closed form with `fit(x, y)`; also has `save(dictfile)`, `load(dictfile)`, `copy(x)`. |
| `Sin` | `Sin()` | `sin(x)` as an activation module. |
| `SwiGLU` | `SwiGLU(in_features, hidden_features)` | SwiGLU activation block. |

### Losses and objectives

| object | signature |
|---|---|
| `average_quadratic_loss` | `average_quadratic_loss(f, y)` |
| `average_binary_cross_entropy_loss` | `average_binary_cross_entropy_loss(f, y)` |
| `ExponentialLoss` | `ExponentialLoss()` |
| `Objective` | `Objective(model, avgloss)` — wraps a model and an average-loss function; `eval()`, `train()`, `save(paramsfile)`, `forward(x, y)` |
| `compute_avg_loss` | `compute_avg_loss(objective, loader)` |

### Training helpers

| object | signature | notes |
|---|---|---|
| `LRStepScheduler` | `LRStepScheduler(optimizer, n_steps, n_iters_per_step, base_lr, gamma, verbose=True)` | Step learning-rate schedule; `step()`, `lr()`. |
| `number_of_parameters` | `number_of_parameters(model)` | Number of trainable parameters. |
| `initialize_model` | `initialize_model(model, paramsfile)` | Load parameters into a model. |
| `initialize_parameters` | `initialize_parameters(model)` | (Re)initialize a model's parameters. |

### `Config` — run configuration

```python
cfg = drwho.nn.Config(name, dirname=None, dirpath=None, mkdir=True, verbose=0)
```

`Config` manages a small YAML-backed dictionary of settings and the standard file
names for a run. It is constructed in one of two ways:

1. **From a name stub** — creates a new configuration and a log directory
   `runs/<dirname>/` (prefixed by `dirpath` if given; `dirname` defaults to a
   timestamp). Standard file names are derived from the stub.
2. **From a `.yaml`/`.yml` file name** — loads that file and rewrites the
   standard file names to sit alongside it.

The standard names live under the `file` key:

| key | value |
|---|---|
| `file/losses` | `<logdir><name>_losses.csv` |
| `file/params` | `<logdir><name>_params.pth` |
| `file/init_params` | `<logdir><name>_init_params.pth` |
| `file/plots` | `<logdir><name>_plots.png` |
| `file/config` | `<logdir><name>_config.yaml`, or the file that was loaded |

> Removed in 0.3.0: `file/script`, which named a TorchScript archive. TorchScript
> serialization is deprecated in PyTorch; save the state dict to `file/params`
> and rebuild the model from its class. A configuration written by an earlier
> version has the key dropped when it is loaded.

Values are read and written through the call operator, with `/` separating
levels; a missing key is created when a value is supplied:

```python
cfg('batchsize', 512)        # create or update, returns 512
cfg('batchsize')             # read
cfg('file/params')           # nested read
cfg.save()                   # write to file/config
```

`configname(name, dirname)` builds a configuration file name from the same
convention.

---

## `data.py` — datasets and loading

| object | signature | notes |
|---|---|---|
| `Dataset` | `Dataset(data, start, end, targets=None, split_col=None, requires_grad=False, random_sample_size=None, device=..., verbose=1)` | Wraps a dataframe slice as tensors on the chosen device (CUDA when available). |
| `DataLoader` | `DataLoader(dataset, batch_size, num_iterations=None, verbose=1, debug=0, shuffle=False)` | A loader substantially faster than PyTorch's default for in-memory data. |
| `SobolSample` | — | Low-discrepancy (Sobol) sampling. |
| `UniformSample` | — | Uniform sampling. |
| `download` | `download(datafile, website='http://www.hep.fsu.edu/~harry/datasets', timeout=10)` | Fetch a dataset from the course data area. |

---

## `monitor.py` — watching training

| object | signature | notes |
|---|---|---|
| `Monitor` | `Monitor(niterations, lossfile, monitorstep, newlossfile=True, frac=0.005, model=None, paramsfile=None, use_tensorboard=False, ylabel=None)` | Writes train/validation losses to a CSV and displays them live in a separate window. Call it once per monitored step. Also `step()`, `reset()`, `start()`, `read_checkpoint()`, `terminate(delay=5)`. |
| `LossMonitor` | `LossMonitor(lossfile, ylabel='$R(\omega)$', ylog=True, xlog=False)` | Standalone viewer for an existing loss file; `show()`. This is what `monlosses` drives. |
| `TimeLeft` | `TimeLeft(N)` | Estimated time remaining over `N` iterations; call with the current index. |
| `get_losses` | `get_losses(loss_file)` | Read a loss CSV. |
| `get_timeleft` | `get_timeleft(timeleft_file)` | Read the time-left file. |
| `elapsed_time` | `elapsed_time(now, start)` | Formatted elapsed time. |
| `plot_loss_curve` | `plot_loss_curve(losses)` | Plot a loss curve. |
| `tensorboard_available` | `tensorboard_available()` | `True` if tensorboard can be imported. |

TensorBoard is **optional**. It is imported only when a `Monitor` is created with
`use_tensorboard=True`, and if it is missing the builtin monitor is used instead
with a one-line warning. Install it with:

```bash
pip install "drwho[tensorboard] @ git+https://github.com/hbprosper/drwho.git@main"
```

### The `monlosses` command

Installed as a console script; watches a loss file written by `Monitor`.

```bash
monlosses loss-file [ylabel] [ylog=1]
```

---

## `kdtree.py` — KD-tree

Recursive binary partitioning of *n* points in *m* dimensions.

| object | signature | notes |
|---|---|---|
| `KDTree` | `KDTree(points, leaf_size=1, store_nodeinfo=False)` | Build the tree. `find_leaf(point)`, `get_leaves()`, `get_nodeinfo()`, `number_of_leaves()`. |
| `KDNode` | `KDNode(points, indices, axis, bounds, left=None, right=None)` | A node, i.e. a bin. A node with no children is a leaf, carrying `ID`, `points`, `indices`. `is_leaf()`. |

---

## Conventions for contributors (i.e. me)

- **Bump `__version__` in `__init__.py` on every release.** `pyproject.toml`
  reads it from there, and pip decides whether to reinstall by comparing version
  strings — not commits. A change pushed without a bump reaches nobody.
- **Add new submodules to `__all__`** in `__init__.py`, or lazy loading will not
  find them.
- **Do not pin `torch` or `numpy`.** Colab and the GPU cluster ship builds matched
  to their own drivers; an upper bound forces a multi-gigabyte reinstall at the
  worst possible moment.
- **Keep optional dependencies lazy.** Probe them where they are used, not at
  module import, so that everyone who imports the module is not told about
  something they did not ask for.
- **Keep the repository lean.** No data files, no notebooks with stored outputs —
  it is cloned afresh on every Colab runtime.
- **Nothing project-specific goes here.** If it mentions a particular analysis by
  name, it belongs in that project's package.
