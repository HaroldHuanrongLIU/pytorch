# Custom Python Process Groups (Experimental)

> **Note:** This is an experimental feature and subject to change.

## Overview

Distributed training code is full of calls like `dist.all_reduce`,
`dist.broadcast`, and `dist.barrier`.  When end users need to add cross-cutting
behavior -- logging every collective, timing them, compressing tensors before
sending, or injecting fault tolerance -- they typically have to modify every
call site throughout their codebase. This is tedious, error-prone, and hard to
maintain.

Custom Python process groups solve this by letting you customize behavior at the
process group level, not the call site level.  All of the custom logic sits in a
centralized location -- the custom PG class and its creator function. The only
changes at the user callsite are:

1. Register the custom PG backend (one-time setup).
2. Update the backend string in `dist.init_process_group`.

After that, every `dist.*` call in the entire codebase automatically goes
through the custom logic.  No other code changes needed.

Here are some common use cases:

**Logging and monitoring.** You want to log every collective operation, track
latency, count bytes transferred, or emit metrics to a dashboard. With a
passthrough PG, you override the collectives you care about, add your logging,
and delegate to the inner PG. Every `dist.*` call in your training loop is
automatically instrumented.

**Timing and profiling.** You want to measure how long each collective takes
without modifying your training code.  A passthrough PG can wrap each
collective's Work object to record start/end times, then return the wrapped Work
to the caller transparently.

**Compression and quantization.** You want to compress tensors
before sending and decompress after receiving to reduce
communication bandwidth. A passthrough PG overrides `all_gather`
or `all_to_all` to compress the input, call the inner PG's
collective, and decompress the output.

**Custom collective implementations.** You have a specialized implementation of
a subset of collectives optimized for your infrastructure -- for example, a
hierarchical all-reduce that exploits your network topology. You can use either
a passthrough PG (overriding specific collectives while delegating the rest to
NCCL) or a terminal PG (implementing everything from scratch).

**Custom collective functions.** You want to define entirely new collective
operations like `my_special_all_reduce` that don't exist in PyTorch. You can
define these as methods on your custom PG and call them as
`dist.my_special_all_reduce(tensor, group=pg)` without modifying PyTorch. The
module-level `__getattr__` on `torch.distributed` forwards unknown function
names to the PG automatically.

**Composing multiple customizations.** You want logging AND compression, each as
a separate reusable module.  Passthrough PGs can be stacked:
`"logging(compression(cuda:nccl))"`.  Each layer handles its concern and
delegates to the next.  The layers are independent and can be mixed and matched.

The key idea: a custom PG intercepts `dist.*` calls at the Python level and
receives the original Python arguments.  It can implement collectives from
scratch (terminal PG), selectively override some and delegate the rest
(passthrough PG), or stack multiple layers of custom behavior on top of a
standard backend.

## Quick Start

There are three steps to creating a custom PG:

1. Define a class that extends `ProcessGroup` (terminal) or
   `PassthroughProcessGroup` (passthrough).
2. Define a creator function that constructs the PG.
3. Register the backend with `dist.Backend.register_backend`.

Then use it with `dist.init_process_group("my_backend")` or, for passthrough
backends, `dist.init_process_group("my_backend(cuda:nccl)")`.

## Terminal Process Groups

A terminal PG extends `ProcessGroup` directly and implements
collectives from scratch.  To intercept `dist.*` calls like
`dist.all_reduce(tensor)`, define the method with the Python
dist.* API signature (e.g., `all_reduce(self, tensor, op,
async_op)`).  You can also override C++ virtual methods (e.g.,
`allreduce(self, tensor_list, opts)`) for code that calls them
directly on the PG instance.

```python
import torch.distributed as dist

class MyPG(dist.ProcessGroup):
    def __init__(self, rank, size):
        super().__init__(rank, size)

    def getBackendName(self):
        return "my_pg"

    def all_reduce(self, tensor, op=None, async_op=False):
        # custom implementation
        ...

def create_my_pg(dist_opts, pg_options=None):
    return MyPG(dist_opts.group_rank, dist_opts.group_size)

dist.Backend.register_backend(
    "my_pg", create_my_pg, extended_api=True,
)
dist.init_process_group("my_pg")
```

After `init_process_group`, calls like `dist.all_reduce(tensor)` are forwarded
directly to `MyPG.all_reduce` with the original Python arguments. Collectives
not defined on `MyPG` fall through to the standard C++ path (which will fail if
there is no C++ backend).

## Passthrough Process Groups

A passthrough PG wraps an inner PG, intercepts some collectives, and delegates
everything else to the inner PG.  This is the common pattern for adding
monitoring, compression, quantization, or other middleware on top of a standard
backend like NCCL.

```python
import torch.distributed as dist

class MyWrapper(dist.PassthroughProcessGroup):
    def all_reduce(self, tensor, op=None, async_op=False, **kwargs):
        if self._should_customize(tensor):
            # custom path
            ...
        else:
            # fall back to the inner PG
            return self._inner_pg.all_reduce(
                tensor, op=op, async_op=async_op, **kwargs,
            )

def create(dist_opts, pg_options=None):
    pg = MyWrapper(dist_opts.group_rank, dist_opts.group_size)
    dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
    return pg

dist.Backend.register_backend(
    "my_wrapper", create, extended_api=True,
)
dist.init_process_group("my_wrapper(cuda:nccl,cpu:gloo)")
```

Key points:

- Call `setup_inner_pg(pg, dist_opts)` in the creator to create and
  attach the inner PG.  Do NOT set `_inner_pg` directly.
- Methods that accept a subset of the `dist.*` parameters should
  include `**kwargs` so unhandled parameters are forwarded.
- `PassthroughProcessGroup` provides default forwarding methods for
  ALL `dist.*` collectives, so you only override what you need.
- C++ virtual methods (`allreduce`, `allgather`, etc.) are also
  forwarded to `self._inner_pg` for code that calls them directly.
- Custom collectives not in the `dist.*` API are delegated via
  `__getattr__`.
- Custom PGs can override `new_group` and `split_group` to
  control how sub-process-groups are created. Without this,
  subgroups created via `dist.new_group()` or
  `dist.split_group()` will be standard ProcessGroup instances
  that do NOT have the custom PG's overrides. If the custom
  behavior should apply to subgroups too, the custom PG must
  override `new_group` and/or `split_group` to wrap the
  resulting subgroup in another instance of itself.

## Nesting / Stacking

Passthrough backends can be stacked using nested backend strings. The nesting
syntax uses parentheses to indicate which backend wraps which:

```python
dist.init_process_group("outer(inner(cuda:nccl,cpu:gloo))")
```

This creates three layers:
- `outer` wraps `inner`, which wraps the standard NCCL/Gloo PG.
- When `dist.broadcast` is called, `outer` gets first shot.  If it
  doesn't override `broadcast`, `PassthroughProcessGroup`'s default
  forwarding calls `dist.broadcast(group=inner_pg)`, which gives
  `inner` a shot, and so on down to NCCL.

Each layer's creator receives `dist_opts.inner` with the next level's
spec.  `setup_inner_pg(pg, dist_opts)` handles the recursion automatically.

The nesting is parsed left-to-right.  The outermost backend
name comes first, followed by its inner in parentheses.
The inner is opaque to PyTorch -- it is simply passed as
a string to the custom process group via `dist_opts.inner`.
The custom process group decides how to interpret it. Examples:

```
"my_logger(cuda:nccl)"
    -- inner "cuda:nccl" passed to my_logger

"compressor(my_logger(cuda:nccl))"
    -- inner "my_logger(cuda:nccl)" passed to compressor

"my_custom_backend"
    -- no inner (dist_opts.inner is None)

"my_custom_backend(cuda,cpu)"
    -- inner "cuda,cpu" passed to my_custom_backend

"my_custom_backend(cuda:nccl,cpu:gloo)"
    -- inner "cuda:nccl,cpu:gloo" passed to
       my_custom_backend
```

Each passthrough creator function looks the same regardless of nesting depth.
The creator does not need to know or care what the inner backend is -- it just
calls `setup_inner_pg(pg, dist_opts)` and the framework handles the rest:

```python
def create_my_logger(dist_opts, pg_options=None):
    pg = MyLoggerPG(dist_opts.group_rank, dist_opts.group_size)
    dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
    return pg

def create_compressor(dist_opts, pg_options=None):
    pg = CompressorPG(dist_opts.group_rank, dist_opts.group_size)
    dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
    return pg
```

Both creators are identical in structure.  To compose them:

```python
dist.Backend.register_backend(
    "my_logger", create_my_logger, extended_api=True, )
dist.Backend.register_backend(
    "compressor", create_compressor, extended_api=True, )

# Use them individually:
dist.init_process_group("my_logger(cuda:nccl)")

# Or stack them:
dist.init_process_group("compressor(my_logger(cuda:nccl))")
```

The layers are independent and reusable.  You can add, remove, or reorder them
by changing only the backend string.

## Custom Work Objects

Custom PGs can return their own Work objects from collective methods for
monitoring, logging, or custom async behavior:

```python
class MonitoredWork:
    def __init__(self, inner_work, op_name):
        self._inner = inner_work
        self._op_name = op_name

    def wait(self):
        start = time.time()
        result = self._inner.wait() if self._inner else True
        log_latency(self._op_name, time.time() - start)
        return result

class MonitoredPG(dist.PassthroughProcessGroup):
    def all_reduce(self, tensor, op=None, async_op=False, **kwargs):
        work = self._inner_pg.all_reduce(
            tensor, op=op, async_op=async_op, **kwargs,
        )
        return MonitoredWork(work, "all_reduce")
```

The caller receives the custom Work object and `.wait()` works transparently.
This enables per-collective latency tracking, profiling, or custom
synchronization behavior.

## pg_options

`pg_options` can be passed as a dict keyed by backend name to provide per-layer
options:

```python
dist.init_process_group(
    "outer(inner(cuda:nccl,cpu:gloo))",
    pg_options={
        "outer": OuterOpts(...),
        "inner": InnerOpts(...),
        "dist": ProcessGroupNCCL.Options(...),
    }, )
```

- Each layer's creator receives its own entry as the `pg_options`
  argument.  For example, `outer`'s creator gets `OuterOpts(...)`.
- The `"dist"` entry carries terminal C++ backend options (e.g.,
  `ProcessGroupNCCL.Options`) and is available to every layer via
  `dist_opts.pg_options`.
- Keys must be either registered backend names or `"dist"`. Unknown
  keys raise `ValueError`.
- If `pg_options` is not a dict, it is treated as dist/terminal
  options: each custom layer's creator receives `pg_options=None`,
  and the value passes through to the C++ backend at the bottom.

## File Organization

- `torch/distributed/custom_pg.py`: `PassthroughProcessGroup`,
  `setup_inner_pg`, `_pg_bypass`, and internal helpers.
- `torch/distributed/CUSTOM_PG.md`: this documentation.
- `torch/distributed/distributed_c10d.py`: imports from
  `custom_pg.py`, `@_pg_bypass`-decorated `dist.*` functions,
  `_new_process_group_helper` integration for nested backend strings.
- `test/distributed/test_custom_pg.py`: all custom PG tests.

## Implementation Details

- `@_pg_bypass` decorator on `dist.*` collective functions walks
  the MRO (stopping at `ProcessGroup`) to find methods with
  matching parameter names.  The signature check (subset match,
  ignoring `*args`/`**kwargs`) distinguishes `dist.*` API overrides
  from C++ virtual method overrides.  Results are cached per class.
- `PassthroughProcessGroup` forwarding methods call
  `self._dist.<fn>(group=self._inner_pg)`, re-entering the
  `_pg_bypass` mechanism on the inner PG.  This is how collectives
  chain through nested layers without explicit inner-PG walking.
- `_parse_nested_backend()` parses `"outer(inner(...))"` backend
  strings into `(outermost, inner)` tuples.
- `_DistributedBackendOpts` wraps the C++
  `_DistributedBackendOptions` to add `inner`, `pg_options`,
  and `_remaining_pg_options`.
- `_pop_pg_options()` splits a `pg_options` dict into per-layer,
  dist, and remaining entries.
- `_create_process_group()` uses a unique inner group name to avoid
  registration conflicts with the outer PG.
- Module-level `__getattr__` on `torch.distributed` forwards custom
  collectives defined on PG subclasses.  Guarded by
  `is_initialized()` so submodule imports (e.g.,
  `torch.distributed.fsdp`) are not intercepted during startup.
