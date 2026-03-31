# niftyreg-python

[English version](./README.md)

`niftyreg-python` 是以 `pybind11 + scikit-build-core` 封裝的 NiftyReg Python 介面，主要對接 `SimpleITK.Image`，提供常用的配準與形變操作。

目前包含：

- `aladin(...)`：剛性 / 仿射配準
- `f3d(...)`：非剛性配準（含 velocity 模式）
- `resample / compose / invert / jacobian / measure / average_affines`
- 以 `SimpleITK.Image` 作為主要輸入輸出型別
- affine 結果以 `LPS` 世界座標回傳（符合 SimpleITK 慣例）

---

## 1. 安裝需求

### 系統需求

- Linux（建議 Ubuntu 20.04+）
- C++17 編譯器（GCC/Clang）
- CMake（建議 3.22+）
- Ninja（建議）
- Python 3.10+

### Python 依賴

由 `pyproject.toml` 定義：

- `numpy>=1.26`
- `SimpleITK>=2.0`
- 建置時需要：`scikit-build-core`、`pybind11`

---

## 2. 快速安裝（開發模式）

> 專案預設使用 `uv` 進行 Python 環境與套件管理。

在 repo 根目錄執行：

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
uv pip install -e .
```

安裝完成後可檢查：

```bash
python -c "import niftyreg; print('niftyreg import OK')"
```

---

## 3. 建置說明（CPU / GPU）

本專案使用 `scikit-build-core` 觸發 CMake 建置，預設為 **CPU build**。

### 3.1 CPU 建置（預設）

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3.2 啟用 CUDA 建置（GPU）

如果要編譯 CUDA 版本，安裝前加入 CMake 參數：

```bash
cd niftyreg-python
uv venv
source .venv/bin/activate
CMAKE_ARGS="-DNIFTYREG_PYTHON_USE_CUDA=ON -DNIFTYREG_PYTHON_CHECK_GPU=ON" uv pip install -e .
```

說明：

- `NIFTYREG_PYTHON_USE_CUDA=ON`：啟用 NiftyReg CUDA backend 編譯
- `NIFTYREG_PYTHON_CHECK_GPU=ON`：啟用 GPU 可用性檢查

執行時再透過 `options.use_cuda=True` 選擇 GPU 路徑（見下方使用範例）。

### 3.3 常用建置環境變數

- `CMAKE_ARGS`：傳遞 CMake 選項（例如 CUDA 開關）
- `CMAKE_BUILD_PARALLEL_LEVEL`：設定平行編譯執行緒數

範例：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install -e .
```

---

## 4. 使用方式

### 4.1 基本：剛性/仿射配準（aladin）

```python
import SimpleITK as sitk
import niftyreg

fixed = sitk.ReadImage("fixed.nii.gz", sitk.sitkFloat32)
moving = sitk.ReadImage("moving.nii.gz", sitk.sitkFloat32)

result = niftyreg.aladin(fixed, moving)
sitk.WriteImage(result.warped, "warped_aladin.nii.gz")
print(result.affine_lps)  # 4x4 LPS affine matrix
```

### 4.2 基本：非剛性配準（f3d）

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
        use_cuda=False,  # 若已用 CUDA 編譯可改成 True
    ),
)

sitk.WriteImage(result.warped, "warped_f3d.nii.gz")
# result.cpp 為控制點網格（Transform）
```

### 4.3 Transform / Resample / Jacobian / Measure

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

## 5. 範例腳本

範例位於 `examples/`：

- `examples/aladin_basic.py`
- `examples/f3d_basic.py`
- `examples/f3d_gpu_fixed_moving.py`
- `examples/rigid_then_f3d_sitk.py`
- `examples/transform_ops.py`

在 repo 根目錄執行：

```bash
source niftyreg-python/.venv/bin/activate
python niftyreg-python/examples/aladin_basic.py
python niftyreg-python/examples/f3d_basic.py
python niftyreg-python/examples/rigid_then_f3d_sitk.py
python niftyreg-python/examples/transform_ops.py
python niftyreg-python/examples/f3d_gpu_fixed_moving.py
```

---

## 6. 測試

安裝測試套件並執行：

```bash
cd niftyreg-python
source .venv/bin/activate
uv pip install -e ".[test]"
pytest tests
```

---

## 7. 常見問題

### Q1: `import niftyreg` 失敗

- 確認已啟用正確虛擬環境：`source niftyreg-python/.venv/bin/activate`
- 重新安裝：`uv pip install -e .`

### Q2: 設定 `use_cuda=True` 但沒有走 GPU

- 先確認建置時是否有帶 `-DNIFTYREG_PYTHON_USE_CUDA=ON`
- 確認機器有可用 NVIDIA 驅動與 CUDA runtime
- 先用 `verbose=True` 觀察執行輸出

---

## 8. API 入口

主要可用符號（`python/niftyreg/__init__.py`）：

- `aladin`, `AladinOptions`
- `f3d`, `F3DOptions`
- `resample`, `compose_transforms`, `invert_transform`
- `transform_to_deformation`, `transform_to_displacement`, `transform_to_flow`
- `jacobian`, `measure`
- `average_affines`, `average_images`

---

## 9. 參考資料

- NiftyReg 官方專案：<https://github.com/KCL-BMEIS/niftyreg>
- Ourselin et al. (2001). Reconstructing a 3D structure from serial histological sections. *Image and Vision Computing*, 19(1-2), 25-31.
- Modat et al. (2014). Global image registration using a symmetric block-matching approach. *Journal of Medical Imaging*, 1(2), 024003.
- Rueckert et al. (1999). Nonrigid registration using free-form deformations: Application to breast MR images. *IEEE Transactions on Medical Imaging*, 18(8), 712-721.
- Modat et al. (2010). Fast free-form deformation using graphics processing units. *Computer Methods and Programs in Biomedicine*, 98(3), 278-284.
