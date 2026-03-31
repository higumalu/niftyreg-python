# niftyreg-python

[繁體中文版本 (Traditional Chinese)](./README.zh-TW.md)

`niftyreg-python` is a `pybind11 + scikit-build-core` Python interface for a subset of NiftyReg, designed around `SimpleITK.Image` I/O for registration and transform workflows.

Current scope:

- `aladin(...)` for rigid / affine registration
- `f3d(...)` for non-rigid registration (including velocity mode)
- `resample / compose / invert / jacobian / measure / average_affines`
- `SimpleITK.Image` as primary input/output type
- affine result returned in `LPS` world coordinates (SimpleITK convention)

---

## 1. Requirements

### System requirements

- Linux (Ubuntu 20.04+ recommended)
- C++17 compiler (GCC/Clang)
- CMake (3.22+ recommended)
- Ninja (recommended)
- Python 3.10+

### Python dependencies

Defined in `pyproject.toml`:

- `numpy>=1.26`
- `SimpleITK>=2.0`
- build-time: `scikit-build-core`, `pybind11`

---

## 2. Quick install (development mode)

> This project uses `uv` as the default Python environment/package manager.

From repo root:

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
uv pip install -e .
```

Verify installation:

```bash
python -c "import niftyreg; print('niftyreg import OK')"
```

---

## 3. Build guide (CPU / GPU)

This project is built by CMake through `scikit-build-core`. Default build is **CPU**.

### 3.1 CPU build (default)

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3.2 CUDA build (GPU)

To compile with CUDA, pass CMake options before install:

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
CMAKE_ARGS="-DNIFTYREG_PYTHON_USE_CUDA=ON -DNIFTYREG_PYTHON_CHECK_GPU=ON" uv pip install -e .
```

Flags:

- `NIFTYREG_PYTHON_USE_CUDA=ON`: enable CUDA backend build
- `NIFTYREG_PYTHON_CHECK_GPU=ON`: enable GPU availability checks

At runtime, set `options.use_cuda=True` to use the GPU path.

### 3.3 Useful build environment variables

- `CMAKE_ARGS`: pass CMake options
- `CMAKE_BUILD_PARALLEL_LEVEL`: control parallel build jobs

Example:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install -e .
```

---

## 4. Usage

### 4.1 Basic rigid/affine registration (`aladin`)

```python
import SimpleITK as sitk
import niftyreg

fixed = sitk.ReadImage("fixed.nii.gz", sitk.sitkFloat32)
moving = sitk.ReadImage("moving.nii.gz", sitk.sitkFloat32)

result = niftyreg.aladin(fixed, moving)
sitk.WriteImage(result.warped, "warped_aladin.nii.gz")
print(result.affine_lps)  # 4x4 LPS affine matrix
```

### 4.2 Basic non-rigid registration (`f3d`)

```python
import SimpleITK as sitk
import niftyreg

fixed = sitk.ReadImage("fixed.nii.gz", sitk.sitkFloat32)
moving = sitk.ReadImage("moving.nii.gz", sitk.sitkFloat32)

result = niftyreg.f3d(
    fixed,
    moving,
    options=niftyreg.F3DOptions(
        use_velocity=False,
        use_symmetric=True,
        use_cuda=False,  # switch to True for CUDA build/runtime
    ),
)

sitk.WriteImage(result.warped, "warped_f3d.nii.gz")
# result.cpp is the control-point grid transform
```

### 4.3 Transform / resample / jacobian / measure

```python
import niftyreg

# result = niftyreg.f3d(...)
deformation = niftyreg.transform_to_deformation(fixed, result.cpp)
warped = niftyreg.resample(fixed, moving, deformation, interpolation=3, padding_value=0.0)
jac = niftyreg.jacobian(fixed, deformation, determinant=True, log_determinant=True)
scores = niftyreg.measure(fixed, warped, metrics=("nmi", "lncc", "ssd"))
print(scores)
```

---

## 5. Example scripts

Examples in `examples/`:

- `examples/aladin_basic.py`
- `examples/f3d_basic.py`
- `examples/f3d_gpu_fixed_moving.py`
- `examples/rigid_then_f3d_sitk.py`
- `examples/transform_ops.py`

Run from repo root:

```bash
source niftyreg-python/.venv/bin/activate
python niftyreg-python/examples/aladin_basic.py
python niftyreg-python/examples/f3d_basic.py
python niftyreg-python/examples/rigid_then_f3d_sitk.py
python niftyreg-python/examples/transform_ops.py
python niftyreg-python/examples/f3d_gpu_fixed_moving.py
```

---

## 6. Testing

Install test extras and run:

```bash
cd niftyreg-python
source .venv/bin/activate
uv pip install -e ".[test]"
pytest tests
```

---

## 7. Troubleshooting

### Q1: `import niftyreg` fails

- make sure you activated the correct environment: `source niftyreg-python/.venv/bin/activate`
- reinstall: `uv pip install -e .`

### Q2: `use_cuda=True` but still not using GPU

- confirm build used `-DNIFTYREG_PYTHON_USE_CUDA=ON`
- confirm NVIDIA driver and CUDA runtime are available
- enable `verbose=True` for additional runtime logs

---

## 8. API entry points

Main exported symbols (`python/niftyreg/__init__.py`):

- `aladin`, `AladinOptions`
- `f3d`, `F3DOptions`
- `resample`, `compose_transforms`, `invert_transform`
- `transform_to_deformation`, `transform_to_displacement`, `transform_to_flow`
- `jacobian`, `measure`
- `average_affines`, `average_images`

---

## 9. References

- Official NiftyReg repository: <https://github.com/KCL-BMEIS/niftyreg>
- Ourselin et al. (2001). Reconstructing a 3D structure from serial histological sections. *Image and Vision Computing*, 19(1-2), 25-31.
- Modat et al. (2014). Global image registration using a symmetric block-matching approach. *Journal of Medical Imaging*, 1(2), 024003.
- Rueckert et al. (1999). Nonrigid registration using free-form deformations: Application to breast MR images. *IEEE Transactions on Medical Imaging*, 18(8), 712-721.
- Modat et al. (2010). Fast free-form deformation using graphics processing units. *Computer Methods and Programs in Biomedicine*, 98(3), 278-284.
