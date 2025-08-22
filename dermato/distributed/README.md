# dermato/distributed

Pytorch Distributed reinterpretation of `saliency_overlap.py`

## running it

1. Install `uv`
2. `$ uv sync`
3. `$ ./torchrun_distributed.sh` (first `chmod a+x torchrun_distributed.sh` if needed)

## notes

So far, it's configured for local machine only.
- `NNODES=1` (a server is an node)
- `NPROC=3` (1 process per visible GPU device on node)
- `HOST=localhost`
- `PORT` is arbitrary (as long as it's high enough not to be privileged)
- A few environment variables need to be set for the command call itself and are placed appropriately:
  - `OMP_NUM_THREADS` is configurable. Pytorch defaults to `1` but it's overly cautious.
  - `CUDA_VISIBLE_DEVICES=0,1,3` or similar, to exclude devices in use by others. Script internally adapts.

## maintaining

1. check for dependency updates: `$ uv lock --upgrade`
    - if updated, include `uv.lock` in the git commit