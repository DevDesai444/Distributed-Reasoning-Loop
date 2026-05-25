# Kaggle Training Notebook Cells

Use these edits in `drlll2.ipynb` before starting a clean Kaggle run from the pushed `main` branch.

## Cell 3: Runtime Helpers

Replace the old helper cell with a version that removes `DRL_DISABLE_8BIT`, exposes separate generation/training environments, and can report lingering accelerator workers.

## Cell 6: Dependencies

Add `nvidia-nvjitlink-cu13` to the install command so Kaggle CUDA 13 runtimes expose `libnvJitLink.so.13` to bitsandbytes:

```python
"transformers accelerate datasets trl peft bitsandbytes nvidia-nvjitlink-cu13 "
```

After installing the editable package, run a diagnostics check with:

```python
env = build_runtime_env(training=True)
run("python -m bitsandbytes", cwd=REPO_DIR, env=env, check=False)
```

## Cell 7: Initial GPU Check

Change the environment creation to generation mode:

```python
env = build_runtime_env(training=False)
run("nvidia-smi", cwd=REPO_DIR, env=env, check=False)
```

This keeps bitsandbytes requirements off during generation while preserving the CUDA/vLLM environment setup.
