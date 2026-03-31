#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_reg_aladin.h"
#include "_reg_aladin_sym.h"
#include "_reg_f3d.h"
#include "_reg_f3d2.h"
#include "_reg_globalTrans.h"
#include "_reg_kld.h"
#include "_reg_lncc.h"
#include "_reg_localTrans.h"
#include "_reg_localTrans_jac.h"
#include "_reg_mind.h"
#include "_reg_nmi.h"
#include "_reg_resampling.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "RNifti/NiftiImage.h"

namespace py = pybind11;
using PrecisionType = float;
using RNifti::NiftiImage;

namespace {

struct ImageMeta {
    std::vector<long long> shape_xyz;
    std::vector<double> spacing_xyz;
    std::vector<double> origin_xyz;
    std::vector<double> direction;
    std::array<float, 16> affine_ras {};
};

enum class TransformKind {
    Affine,
    Deformation,
    Displacement,
    Cpp,
    Flow,
    DisplacementFlow,
    SplineVelocity,
};

struct TransformPayload {
    TransformKind kind;
    std::optional<mat44> affine;
    NiftiImage image;
};

ImageMeta parse_meta(const py::dict &meta) {
    ImageMeta parsed;
    parsed.shape_xyz = meta["shape_xyz"].cast<std::vector<long long>>();
    parsed.spacing_xyz = meta["spacing_xyz"].cast<std::vector<double>>();
    parsed.origin_xyz = meta["origin_xyz"].cast<std::vector<double>>();
    parsed.direction = meta["direction"].cast<std::vector<double>>();
    auto affine = meta["affine_ras"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto buf = affine.request();
    if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
        throw std::invalid_argument("affine_ras must be a 4x4 matrix");
    }
    const auto *ptr = static_cast<const double *>(buf.ptr);
    for (size_t i = 0; i < 16; ++i) {
        parsed.affine_ras[i] = static_cast<float>(ptr[i]);
    }
    return parsed;
}

py::dict meta_to_py(const NiftiImage &image) {
    py::dict meta;
    const int spatial_dims = image->nz > 1 ? 3 : 2;
    mat44 affine_lps = image->qto_xyz;
    affine_lps.m[0][0] *= -1.0f;
    affine_lps.m[0][1] *= -1.0f;
    affine_lps.m[0][2] *= -1.0f;
    affine_lps.m[0][3] *= -1.0f;
    affine_lps.m[1][0] *= -1.0f;
    affine_lps.m[1][1] *= -1.0f;
    affine_lps.m[1][2] *= -1.0f;
    affine_lps.m[1][3] *= -1.0f;
    py::array_t<long long> shape({spatial_dims});
    py::array_t<double> spacing({spatial_dims});
    py::array_t<double> origin({spatial_dims});
    py::array_t<double> direction({spatial_dims * spatial_dims});
    auto shape_buf = shape.mutable_unchecked<1>();
    auto spacing_buf = spacing.mutable_unchecked<1>();
    auto origin_buf = origin.mutable_unchecked<1>();
    auto direction_buf = direction.mutable_unchecked<1>();
    for (int i = 0; i < spatial_dims; ++i) {
        shape_buf(i) = image->dim[i + 1];
        spacing_buf(i) = image->pixdim[i + 1];
        origin_buf(i) = affine_lps.m[i][3];
        for (int j = 0; j < spatial_dims; ++j) {
            const double spacing_value = image->pixdim[j + 1] == 0.0 ? 1.0 : image->pixdim[j + 1];
            direction_buf(i * spatial_dims + j) = affine_lps.m[i][j] / spacing_value;
        }
    }
    py::array_t<double> affine({4, 4});
    auto affine_buf = affine.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            affine_buf(i, j) = image->qto_xyz.m[i][j];
        }
    }
    meta["shape_xyz"] = shape;
    meta["spacing_xyz"] = spacing;
    meta["origin_xyz"] = origin;
    meta["direction"] = direction;
    meta["affine_ras"] = affine;
    return meta;
}

mat44 to_mat44(const py::array_t<double, py::array::c_style | py::array::forcecast> &array) {
    auto buf = array.request();
    if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
        throw std::invalid_argument("matrix must be 4x4");
    }
    const auto *ptr = static_cast<const double *>(buf.ptr);
    mat44 matrix {};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            matrix.m[row][col] = static_cast<float>(ptr[row * 4 + col]);
        }
    }
    return matrix;
}

py::array_t<double> mat44_to_numpy(const mat44 &matrix) {
    py::array_t<double> result({4, 4});
    auto buf = result.mutable_unchecked<2>();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            buf(row, col) = matrix.m[row][col];
        }
    }
    return result;
}

std::optional<float> optional_float(const py::dict &options, const char *key) {
    if (!options.contains(key) || options[key].is_none()) {
        return std::nullopt;
    }
    return options[key].cast<float>();
}

TransformKind parse_transform_kind(const std::string &kind) {
    if (kind == "affine") return TransformKind::Affine;
    if (kind == "deformation") return TransformKind::Deformation;
    if (kind == "displacement") return TransformKind::Displacement;
    if (kind == "cpp") return TransformKind::Cpp;
    if (kind == "flow") return TransformKind::Flow;
    if (kind == "displacement_flow") return TransformKind::DisplacementFlow;
    if (kind == "spline_velocity") return TransformKind::SplineVelocity;
    throw std::invalid_argument("Unsupported transform kind: " + kind);
}

void apply_geometry(NiftiImage &image, const ImageMeta &meta) {
    const int spatial_dims = static_cast<int>(meta.shape_xyz.size());
    for (int i = 0; i < spatial_dims; ++i) {
        image.setPixDim(static_cast<NiftiImage::Dim>(i + 1), static_cast<float>(meta.spacing_xyz[i]));
    }
    const auto affine = [&]() {
        mat44 matrix {};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                matrix.m[row][col] = meta.affine_ras[row * 4 + col];
            }
        }
        return matrix;
    }();
    image->qform_code = NIFTI_XFORM_SCANNER_ANAT;
    image->sform_code = NIFTI_XFORM_SCANNER_ANAT;
    image->qto_xyz = affine;
    image->qto_ijk = nifti_mat44_inverse(affine);
    image->sto_xyz = affine;
    image->sto_ijk = nifti_mat44_inverse(affine);

    float qb = 0.0f, qc = 0.0f, qd = 0.0f;
    float qx = 0.0f, qy = 0.0f, qz = 0.0f;
    float dx = 1.0f, dy = 1.0f, dz = 1.0f, qfac = 1.0f;
    nifti_mat44_to_quatern(affine, &qb, &qc, &qd, &qx, &qy, &qz, &dx, &dy, &dz, &qfac);
    image->quatern_b = qb;
    image->quatern_c = qc;
    image->quatern_d = qd;
    image->qoffset_x = qx;
    image->qoffset_y = qy;
    image->qoffset_z = qz;
    image->qfac = qfac;
}

std::vector<NiftiImage::dim_t> scalar_dims_from_meta(const ImageMeta &meta) {
    std::vector<NiftiImage::dim_t> dims;
    dims.reserve(meta.shape_xyz.size());
    for (const auto value : meta.shape_xyz) {
        dims.push_back(static_cast<NiftiImage::dim_t>(value));
    }
    return dims;
}

std::vector<NiftiImage::dim_t> vector_dims_from_meta(const ImageMeta &meta, int components) {
    std::vector<NiftiImage::dim_t> dims = scalar_dims_from_meta(meta);
    if (meta.shape_xyz.size() == 2) {
        dims.push_back(1);
    }
    dims.push_back(1);
    dims.push_back(components);
    return dims;
}

template <typename T>
NiftiImage make_scalar_image(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    const ImageMeta &meta,
    int datatype
) {
    auto buf = array.request();
    if (buf.ndim != static_cast<py::ssize_t>(meta.shape_xyz.size())) {
        throw std::invalid_argument("Scalar array rank does not match metadata");
    }
    NiftiImage image(scalar_dims_from_meta(meta), datatype);
    apply_geometry(image, meta);
    std::memcpy(image->data, buf.ptr, static_cast<size_t>(image->nvox) * sizeof(T));
    return image;
}

NiftiImage make_vector_image(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &array,
    const ImageMeta &meta
) {
    auto buf = array.request();
    const int spatial_dims = static_cast<int>(meta.shape_xyz.size());
    if (buf.ndim != spatial_dims + 1) {
        throw std::invalid_argument("Vector array rank does not match metadata");
    }
    const int components = static_cast<int>(buf.shape[buf.ndim - 1]);
    NiftiImage image(vector_dims_from_meta(meta, components), DT_FLOAT32);
    apply_geometry(image, meta);
    float *target = static_cast<float *>(image->data);
    const float *source = static_cast<const float *>(buf.ptr);
    const size_t voxel_count = image.nVoxelsPerVolume();
    for (int component = 0; component < components; ++component) {
        for (size_t voxel = 0; voxel < voxel_count; ++voxel) {
            target[component * voxel_count + voxel] = source[voxel * components + component];
        }
    }
    return image;
}

py::array_t<float> nifti_to_numpy(const NiftiImage &image) {
    if (image->datatype != DT_FLOAT32) {
        throw std::runtime_error("Only float32 outputs are currently supported");
    }
    const int spatial_dims = image->nz > 1 ? 3 : 2;
    const int components = image->nu > 1 ? image->nu : (image->nt > 1 ? image->nt : 1);
    if (components == 1) {
        std::vector<py::ssize_t> shape;
        if (spatial_dims == 2) {
            shape.push_back(image->dim[2]);
            shape.push_back(image->dim[1]);
        } else {
            shape.push_back(image->dim[3]);
            shape.push_back(image->dim[2]);
            shape.push_back(image->dim[1]);
        }
        py::array_t<float> out(shape);
        std::memcpy(out.mutable_data(), image->data, static_cast<size_t>(image->nvox) * sizeof(float));
        return out;
    }
    std::vector<py::ssize_t> shape;
    if (spatial_dims == 2) {
        shape.push_back(image->dim[2]);
        shape.push_back(image->dim[1]);
    } else {
        shape.push_back(image->dim[3]);
        shape.push_back(image->dim[2]);
        shape.push_back(image->dim[1]);
    }
    shape.push_back(components);
    py::array_t<float> out(shape);
    auto *target = static_cast<float *>(out.request().ptr);
    const float *source = static_cast<const float *>(image->data);
    const size_t voxel_count = image.nVoxelsPerVolume();
    for (int component = 0; component < components; ++component) {
        for (size_t voxel = 0; voxel < voxel_count; ++voxel) {
            target[voxel * components + component] = source[component * voxel_count + voxel];
        }
    }
    return out;
}

void set_vector_intent(NiftiImage &image, int intent_p1) {
    image->intent_code = NIFTI_INTENT_VECTOR;
    std::memset(image->intent_name, 0, 16);
    std::strcpy(image->intent_name, "NREG_TRANS");
    image->intent_p1 = intent_p1;
    image->scl_slope = 1.0f;
    image->scl_inter = 0.0f;
}

NiftiImage create_reference_field(const ImageMeta &meta, int datatype = DT_FLOAT32, int components = -1) {
    const int spatial_dims = static_cast<int>(meta.shape_xyz.size());
    if (components < 0) {
        components = spatial_dims;
    }
    NiftiImage image(vector_dims_from_meta(meta, components), datatype);
    apply_geometry(image, meta);
    set_vector_intent(image, DEF_FIELD);
    return image;
}

NiftiImage create_identity_deformation(const ImageMeta &meta) {
    NiftiImage image = create_reference_field(meta);
    std::memset(image->data, 0, static_cast<size_t>(image->nvox) * image->nbyper);
    image->intent_p1 = DISP_FIELD;
    reg_getDeformationFromDisplacement(image);
    return image;
}

std::unique_ptr<int[]> make_mask_from_image(const NiftiImage &mask_image) {
    auto mask = std::make_unique<int[]>(mask_image.nVoxelsPerVolume());
    if (mask_image->datatype != DT_UINT8) {
        throw std::invalid_argument("Mask image must be uint8");
    }
    const auto *ptr = static_cast<const uint8_t *>(mask_image->data);
    for (size_t i = 0; i < mask_image.nVoxelsPerVolume(); ++i) {
        mask[i] = ptr[i] > 0 ? static_cast<int>(i) : -1;
    }
    return mask;
}

TransformPayload parse_transform_payload(const py::dict &payload) {
    TransformPayload transform;
    transform.kind = parse_transform_kind(payload["kind"].cast<std::string>());
    if (transform.kind == TransformKind::Affine) {
        transform.affine = to_mat44(payload["matrix_ras"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
        return transform;
    }
    const auto meta = parse_meta(payload["meta"].cast<py::dict>());
    const auto array = payload["array"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    transform.image = make_vector_image(array, meta);
    switch (transform.kind) {
    case TransformKind::Deformation: set_vector_intent(transform.image, DEF_FIELD); break;
    case TransformKind::Displacement: set_vector_intent(transform.image, DISP_FIELD); break;
    case TransformKind::Cpp: set_vector_intent(transform.image, CUB_SPLINE_GRID); break;
    case TransformKind::Flow: set_vector_intent(transform.image, DEF_VEL_FIELD); break;
    case TransformKind::DisplacementFlow: set_vector_intent(transform.image, DISP_VEL_FIELD); break;
    case TransformKind::SplineVelocity: set_vector_intent(transform.image, SPLINE_VEL_GRID); break;
    default: break;
    }
    return transform;
}

NiftiImage copy_image_info_and_alloc(const NiftiImage &image) {
    return NiftiImage(image, NiftiImage::Copy::ImageInfoAndAllocData);
}

NiftiImage ensure_deformation_field(TransformPayload transform, const std::optional<ImageMeta> &reference_meta) {
    switch (transform.kind) {
    case TransformKind::Affine: {
        if (!reference_meta || !transform.affine) {
            throw std::invalid_argument("Affine to deformation requires reference metadata");
        }
        NiftiImage out = create_reference_field(*reference_meta);
        auto affine = *transform.affine;
        reg_affine_getDeformationField(&affine, out);
        out->intent_p1 = DEF_FIELD;
        return out;
    }
    case TransformKind::Deformation:
        return NiftiImage(transform.image, NiftiImage::Copy::Image);
    case TransformKind::Displacement: {
        NiftiImage out(transform.image, NiftiImage::Copy::Image);
        reg_getDeformationFromDisplacement(out);
        return out;
    }
    case TransformKind::Cpp: {
        if (!reference_meta) {
            throw std::invalid_argument("CPP to deformation requires reference metadata");
        }
        NiftiImage out = create_reference_field(*reference_meta);
        reg_spline_getDeformationField(transform.image, out, nullptr, false, true);
        out->intent_p1 = DEF_FIELD;
        return out;
    }
    case TransformKind::SplineVelocity: {
        if (!reference_meta) {
            throw std::invalid_argument("Spline velocity to deformation requires reference metadata");
        }
        NiftiImage out = create_reference_field(*reference_meta);
        reg_spline_getDefFieldFromVelocityGrid(transform.image, out, false);
        out->intent_p1 = DEF_FIELD;
        return out;
    }
    case TransformKind::Flow: {
        NiftiImage out = reference_meta ? create_reference_field(*reference_meta) : copy_image_info_and_alloc(transform.image);
        reg_defField_getDeformationFieldFromFlowField(transform.image, out, false);
        out->intent_p1 = DEF_FIELD;
        return out;
    }
    case TransformKind::DisplacementFlow: {
        NiftiImage out = reference_meta ? create_identity_deformation(*reference_meta) : copy_image_info_and_alloc(transform.image);
        if (!reference_meta) {
            out->intent_p1 = DISP_FIELD;
            reg_getDeformationFromDisplacement(out);
        }
        reg_defField_getDeformationFieldFromFlowField(transform.image, out, false);
        out->intent_p1 = DEF_FIELD;
        return out;
    }
    }
    throw std::invalid_argument("Unsupported transform kind");
}

NiftiImage ensure_flow_field(TransformPayload transform, const std::optional<ImageMeta> &reference_meta) {
    switch (transform.kind) {
    case TransformKind::Flow:
        return NiftiImage(transform.image, NiftiImage::Copy::Image);
    case TransformKind::DisplacementFlow: {
        NiftiImage out(transform.image, NiftiImage::Copy::Image);
        reg_getDisplacementFromDeformation(out);
        out->intent_p1 = DEF_VEL_FIELD;
        return out;
    }
    case TransformKind::SplineVelocity: {
        if (!reference_meta) {
            throw std::invalid_argument("Spline velocity to flow requires reference metadata");
        }
        NiftiImage out = create_reference_field(*reference_meta);
        reg_spline_getFlowFieldFromVelocityGrid(transform.image, out);
        out->intent_p1 = DEF_VEL_FIELD;
        return out;
    }
    default:
        throw std::invalid_argument("Only velocity transforms can be converted to flow");
    }
}

py::dict pack_transform_output(const std::string &kind, const NiftiImage &image) {
    py::dict payload;
    payload["kind"] = kind;
    payload["array"] = nifti_to_numpy(image);
    payload["meta"] = meta_to_py(image);
    return payload;
}

std::unique_ptr<reg_aladin<PrecisionType>> create_aladin_runner(bool symmetric) {
    if (symmetric) return std::make_unique<reg_aladin_sym<PrecisionType>>();
    return std::make_unique<reg_aladin<PrecisionType>>();
}

void configure_aladin_runner(reg_aladin<PrecisionType> &runner, const py::dict &options) {
    runner.SetPerformRigid(true);
    runner.SetPerformAffine(!options["rigid_only"].cast<bool>());
    runner.SetAlignCentre(options["align_centre"].cast<bool>());
    runner.SetAlignCentreMass(options["align_centre_mass"].cast<int>());
    runner.SetMaxIterations(options["max_iterations"].cast<unsigned>());
    runner.SetNumberOfLevels(options["number_of_levels"].cast<unsigned>());
    runner.SetLevelsToPerform(
        options["levels_to_perform"].is_none()
            ? options["number_of_levels"].cast<unsigned>()
            : options["levels_to_perform"].cast<unsigned>()
    );
    runner.SetBlockPercentage(options["block_percentage"].cast<int>());
    runner.SetInlierLts(options["inlier_lts"].cast<int>());
    runner.SetBlockStepSize(options["block_step_size"].cast<int>());
    runner.SetInterpolation(options["interpolation"].cast<int>());
    runner.SetFloatingSigma(options["floating_sigma"].cast<float>());
    runner.SetReferenceSigma(options["reference_sigma"].cast<float>());
    runner.SetWarpedPaddingValue(options["padding_value"].cast<float>());
#if defined(USE_CUDA)
    if (options.contains("use_cuda") && options["use_cuda"].cast<bool>()) {
        runner.SetPlatformType(PlatformType::Cuda);
    } else {
        runner.SetPlatformType(PlatformType::Cpu);
    }
#else
    runner.SetPlatformType(PlatformType::Cpu);
#endif
    runner.SetGpuIdx(999);
    if (options["affine_direct"].cast<bool>()) {
        runner.SetPerformRigid(false);
        runner.SetPerformAffine(true);
    }
    if (const auto value = optional_float(options, "reference_lower_threshold")) runner.SetReferenceLowerThreshold(*value);
    if (const auto value = optional_float(options, "reference_upper_threshold")) runner.SetReferenceUpperThreshold(*value);
    if (const auto value = optional_float(options, "floating_lower_threshold")) runner.SetFloatingLowerThreshold(*value);
    if (const auto value = optional_float(options, "floating_upper_threshold")) runner.SetFloatingUpperThreshold(*value);
    if (runner.GetLevelsToPerform() > runner.GetNumberOfLevels()) {
        runner.SetLevelsToPerform(runner.GetNumberOfLevels());
    }
    runner.SetVerbose(options["verbose"].cast<bool>());
}

std::unique_ptr<reg_f3d<PrecisionType>> create_f3d_runner(bool use_velocity, bool use_symmetric) {
    if (use_velocity || use_symmetric) {
        return std::make_unique<reg_f3d2<PrecisionType>>(1, 1);
    }
    return std::make_unique<reg_f3d<PrecisionType>>(1, 1);
}

void set_floating_mask(reg_aladin<PrecisionType> &runner, NiftiImage &&mask) {
    auto *symmetric_runner = dynamic_cast<reg_aladin_sym<PrecisionType> *>(&runner);
    if (symmetric_runner == nullptr) {
        throw std::invalid_argument("floating_mask requires symmetric=True");
    }
    symmetric_runner->SetInputFloatingMask(std::move(mask));
}

void apply_similarity_options(reg_base<PrecisionType> &runner, const py::dict &options) {
    const std::string similarity = options["similarity"].cast<std::string>();
    if (similarity == "nmi") {
        runner.UseNMISetReferenceBinNumber(0, options["nmi_reference_bins"].cast<int>());
        runner.UseNMISetFloatingBinNumber(0, options["nmi_floating_bins"].cast<int>());
    } else if (similarity == "ssd") {
        runner.UseSSD(0, options["ssd_normalize"].cast<bool>());
    } else if (similarity == "lncc") {
        runner.UseLNCC(0, options["lncc_sigma"].cast<float>());
    } else if (similarity == "mind") {
        runner.UseMIND(0, options["mind_offset"].cast<int>());
    } else if (similarity == "mindssc") {
        runner.UseMINDSSC(0, options["mind_offset"].cast<int>());
    } else if (similarity == "kld") {
        runner.UseKLDivergence(0);
    } else {
        throw std::invalid_argument("Unsupported similarity: " + similarity);
    }
}

py::dict run_aladin_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &reference,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &floating,
    const py::dict &reference_meta_dict,
    const py::dict &floating_meta_dict,
    const py::object &reference_mask_obj,
    const py::object &floating_mask_obj,
    const py::dict &options
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    const ImageMeta floating_meta = parse_meta(floating_meta_dict);
    auto runner = create_aladin_runner(options["symmetric"].cast<bool>());
    runner->SetInputReference(make_scalar_image(reference, reference_meta, DT_FLOAT32));
    runner->SetInputFloating(make_scalar_image(floating, floating_meta, DT_FLOAT32));
    if (!reference_mask_obj.is_none()) {
        runner->SetInputMask(make_scalar_image(reference_mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), reference_meta, DT_UINT8));
    }
    if (!floating_mask_obj.is_none()) {
        set_floating_mask(*runner, make_scalar_image(floating_mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), floating_meta, DT_UINT8));
    }
    configure_aladin_runner(*runner, options);
    runner->Run();
    NiftiImage warped = runner->GetFinalWarpedImage();
    if (warped->datatype != DT_FLOAT32) {
        reg_tools_changeDatatype<float>(warped);
    }
    py::dict result;
    result["warped"] = nifti_to_numpy(warped);
    result["affine_ras"] = mat44_to_numpy(*runner->GetTransformationMatrix());
    return result;
}

py::dict run_aladin_from_files_impl(
    const std::string &reference_path,
    const std::string &floating_path,
    const py::object &reference_mask_path_obj,
    const py::object &floating_mask_path_obj,
    const py::dict &options
) {
    auto runner = create_aladin_runner(options["symmetric"].cast<bool>());
    NiftiImage reference_image = reg_io_ReadImageFile(reference_path.c_str());
    if (!reference_image) {
        throw std::runtime_error("Failed to read reference image: " + reference_path);
    }
    NiftiImage floating_image = reg_io_ReadImageFile(floating_path.c_str());
    if (!floating_image) {
        throw std::runtime_error("Failed to read floating image: " + floating_path);
    }
    runner->SetInputReference(reference_image);
    runner->SetInputFloating(floating_image);
    if (!reference_mask_path_obj.is_none()) {
        const std::string reference_mask_path = reference_mask_path_obj.cast<std::string>();
        NiftiImage reference_mask = reg_io_ReadImageFile(reference_mask_path.c_str());
        if (!reference_mask) {
            throw std::runtime_error("Failed to read reference mask: " + reference_mask_path);
        }
        runner->SetInputMask(reference_mask);
    }
    if (!floating_mask_path_obj.is_none()) {
        const std::string floating_mask_path = floating_mask_path_obj.cast<std::string>();
        NiftiImage floating_mask = reg_io_ReadImageFile(floating_mask_path.c_str());
        if (!floating_mask) {
            throw std::runtime_error("Failed to read floating mask: " + floating_mask_path);
        }
        set_floating_mask(*runner, std::move(floating_mask));
    }
    configure_aladin_runner(*runner, options);
    runner->Run();
    NiftiImage warped = runner->GetFinalWarpedImage();
    if (warped->datatype != DT_FLOAT32) {
        reg_tools_changeDatatype<float>(warped);
    }
    py::dict result;
    result["warped"] = nifti_to_numpy(warped);
    result["affine_ras"] = mat44_to_numpy(*runner->GetTransformationMatrix());
    return result;
}

py::dict run_f3d_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &reference,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &floating,
    const py::dict &reference_meta_dict,
    const py::dict &floating_meta_dict,
    const py::object &reference_mask_obj,
    const py::object &floating_mask_obj,
    const py::object &initial_transform_obj,
    const py::object &initial_cpp_obj,
    const py::dict &options
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    const ImageMeta floating_meta = parse_meta(floating_meta_dict);
    auto runner = create_f3d_runner(options["use_velocity"].cast<bool>(), options["use_symmetric"].cast<bool>());
#if defined(USE_CUDA)
    if (options.contains("use_cuda") && options["use_cuda"].cast<bool>()) {
        runner->SetPlatformType(PlatformType::Cuda);
        if (options["verbose"].cast<bool>()) {
            std::cerr << "[niftyreg_pybind] use_cuda=True -> PlatformType::Cuda" << std::endl;
        }
    }
#endif
    runner->SetReferenceImage(make_scalar_image(reference, reference_meta, DT_FLOAT32));
    runner->SetFloatingImage(make_scalar_image(floating, floating_meta, DT_FLOAT32));
    if (!reference_mask_obj.is_none()) {
        runner->SetReferenceMask(make_scalar_image(reference_mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), reference_meta, DT_UINT8));
    }
    if (!initial_transform_obj.is_none()) {
        const auto transform = parse_transform_payload(initial_transform_obj.cast<py::dict>());
        if (!transform.affine) throw std::invalid_argument("initial_transform must be affine");
        runner->SetAffineTransformation(*transform.affine);
    }
    if (!initial_cpp_obj.is_none()) {
        const auto transform = parse_transform_payload(initial_cpp_obj.cast<py::dict>());
        auto *f3d = dynamic_cast<reg_f3d<PrecisionType> *>(runner.get());
        f3d->SetControlPointGridImage(NiftiImage(transform.image, NiftiImage::Copy::Image));
    }
    if (auto *f3d2 = dynamic_cast<reg_f3d2<PrecisionType> *>(runner.get())) {
        if (!floating_mask_obj.is_none()) {
            f3d2->SetFloatingMask(make_scalar_image(floating_mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), floating_meta, DT_UINT8));
        }
        f3d2->SetInverseConsistencyWeight(options["inverse_consistency_weight"].cast<float>());
        if (options["use_gradient_cumulative_exp"].cast<bool>()) f3d2->UseGradientCumulativeExp();
        else f3d2->DoNotUseGradientCumulativeExp();
        if (!options["bch_terms"].is_none()) f3d2->UseBCHUpdate(options["bch_terms"].cast<int>());
    }
    runner->SetMaximalIterationNumber(options["max_iterations"].cast<unsigned>());
    runner->SetLevelNumber(options["number_of_levels"].cast<unsigned>());
    runner->SetLevelToPerform(options["levels_to_perform"].is_none() ? options["number_of_levels"].cast<unsigned>() : options["levels_to_perform"].cast<unsigned>());
    if (!options["use_pyramid"].cast<bool>()) runner->DoNotUsePyramidalApproach();
    if (options["use_conjugate_gradient"].cast<bool>()) runner->UseConjugateGradient();
    else runner->DoNotUseConjugateGradient();
    if (options["use_approx_gradient"].cast<bool>()) runner->UseApproximatedGradient();
    else runner->DoNotUseApproximatedGradient();
    runner->SetPerturbationNumber(options["perturbation_number"].cast<int>());
    switch (options["interpolation"].cast<int>()) {
    case 0: runner->UseNearestNeighborInterpolation(); break;
    case 1: runner->UseLinearInterpolation(); break;
    default: runner->UseCubicSplineInterpolation(); break;
    }
    runner->SetReferenceSmoothingSigma(options["reference_sigma"].cast<float>());
    runner->SetFloatingSmoothingSigma(options["floating_sigma"].cast<float>());
    runner->SetGradientSmoothingSigma(options["gradient_sigma"].cast<float>());
    runner->SetWarpedPaddingValue(options["padding_value"].cast<float>());
    if (options["robust_range"].cast<bool>()) runner->UseRobustRange();
    else runner->DoNotUseRobustRange();
    if (const auto value = optional_float(options, "reference_lower_threshold")) runner->SetReferenceThresholdLow(0, *value);
    if (const auto value = optional_float(options, "reference_upper_threshold")) runner->SetReferenceThresholdUp(0, *value);
    if (const auto value = optional_float(options, "floating_lower_threshold")) runner->SetFloatingThresholdLow(0, *value);
    if (const auto value = optional_float(options, "floating_upper_threshold")) runner->SetFloatingThresholdUp(0, *value);
    if (options["verbose"].cast<bool>()) runner->PrintOutInformation();
    else runner->DoNotPrintOutInformation();
    apply_similarity_options(*runner, options);
    auto *f3d = dynamic_cast<reg_f3d<PrecisionType> *>(runner.get());
    f3d->SetSpacing(0, options["spacing_x"].cast<float>());
    f3d->SetSpacing(1, options["spacing_y"].is_none() ? options["spacing_x"].cast<float>() : options["spacing_y"].cast<float>());
    f3d->SetSpacing(2, options["spacing_z"].is_none() ? options["spacing_x"].cast<float>() : options["spacing_z"].cast<float>());
    f3d->SetBendingEnergyWeight(options["bending_energy_weight"].cast<float>());
    f3d->SetLinearEnergyWeight(options["linear_energy_weight"].cast<float>());
    f3d->SetJacobianLogWeight(options["jacobian_log_weight"].cast<float>());
    if (options["approximate_jacobian_log"].cast<bool>()) f3d->ApproximateJacobianLog();
    else f3d->DoNotApproximateJacobianLog();
    runner->Run();
    py::dict result;
    result["warped"] = nifti_to_numpy(f3d->GetWarpedImage()[0]);
    const auto cpp = f3d->GetControlPointPositionImage();
    result["cpp"] = nifti_to_numpy(cpp);
    result["cpp_meta"] = meta_to_py(cpp);
    if (auto *f3d2 = dynamic_cast<reg_f3d2<PrecisionType> *>(runner.get())) {
        const auto backward_cpp = f3d2->GetBackwardControlPointPositionImage();
        result["backward_cpp"] = nifti_to_numpy(backward_cpp);
        result["backward_cpp_meta"] = meta_to_py(backward_cpp);
    } else {
        result["backward_cpp"] = py::none();
        result["backward_cpp_meta"] = py::none();
    }
    return result;
}

py::array_t<float> resample_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &reference,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &floating,
    const py::dict &reference_meta_dict,
    const py::dict &floating_meta_dict,
    const py::dict &transform_payload,
    int interpolation,
    float padding_value
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    const ImageMeta floating_meta = parse_meta(floating_meta_dict);
    NiftiImage reference_image = make_scalar_image(reference, reference_meta, DT_FLOAT32);
    NiftiImage floating_image = make_scalar_image(floating, floating_meta, DT_FLOAT32);
    TransformPayload transform = parse_transform_payload(transform_payload);
    NiftiImage deformation = ensure_deformation_field(transform, reference_meta);
    NiftiImage warped(reference_image, NiftiImage::Copy::ImageInfoAndAllocData);
    warped->datatype = floating_image->datatype;
    warped->nbyper = floating_image->nbyper;
    free(warped->data);
    warped->data = std::malloc(static_cast<size_t>(warped->nvox) * floating_image->nbyper);
    reg_resampleImage(floating_image, warped, deformation, nullptr, interpolation, padding_value);
    return nifti_to_numpy(warped);
}

py::dict transform_to_deformation_impl(const py::dict &reference_meta_dict, const py::dict &transform_payload) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    TransformPayload transform = parse_transform_payload(transform_payload);
    const NiftiImage deformation = ensure_deformation_field(transform, reference_meta);
    return pack_transform_output("deformation", deformation);
}

py::dict transform_to_displacement_impl(const py::dict &reference_meta_dict, const py::dict &transform_payload) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    TransformPayload transform = parse_transform_payload(transform_payload);
    NiftiImage displacement = ensure_deformation_field(transform, reference_meta);
    reg_getDisplacementFromDeformation(displacement);
    return pack_transform_output("displacement", displacement);
}

py::dict transform_to_flow_impl(const py::dict &reference_meta_dict, const py::dict &transform_payload) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    TransformPayload transform = parse_transform_payload(transform_payload);
    const NiftiImage flow = ensure_flow_field(transform, reference_meta);
    return pack_transform_output("flow", flow);
}

py::dict compose_transforms_impl(
    const py::dict &reference_meta_dict,
    const py::dict &transform1_payload,
    const py::dict &transform2_payload,
    const py::object &reference2_meta_obj
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    auto transform1 = parse_transform_payload(transform1_payload);
    auto transform2 = parse_transform_payload(transform2_payload);
    NiftiImage output1 = ensure_deformation_field(transform1, reference_meta);
    if (transform2.kind == TransformKind::Affine && transform2.affine) {
        NiftiImage output2(output1, NiftiImage::Copy::ImageInfoAndAllocData);
        set_vector_intent(output2, DEF_FIELD);
        auto affine = *transform2.affine;
        reg_affine_getDeformationField(&affine, output2);
        reg_defField_compose(output2, output1, nullptr);
    } else if (transform2.kind == TransformKind::Cpp) {
        reg_spline_getDeformationField(transform2.image, output1, nullptr, true, true);
    } else {
        std::optional<ImageMeta> reference2_meta;
        if (!reference2_meta_obj.is_none()) {
            reference2_meta = parse_meta(reference2_meta_obj.cast<py::dict>());
        } else {
            reference2_meta = reference_meta;
        }
        NiftiImage output2 = ensure_deformation_field(transform2, reference2_meta);
        reg_defField_compose(output2, output1, nullptr);
    }
    return pack_transform_output("deformation", output1);
}

py::dict invert_transform_impl(const py::dict &reference_meta_dict, const py::dict &transform_payload) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    auto transform = parse_transform_payload(transform_payload);
    py::dict result;
    if (transform.kind == TransformKind::Affine && transform.affine) {
        result["kind"] = "affine";
        result["matrix_ras"] = mat44_to_numpy(nifti_mat44_inverse(*transform.affine));
        return result;
    }
    if (transform.kind == TransformKind::Flow || transform.kind == TransformKind::DisplacementFlow) {
        NiftiImage output = reference_meta.shape_xyz == std::vector<long long>{} ? copy_image_info_and_alloc(transform.image) : create_reference_field(reference_meta);
        output->intent_p1 = transform.image->intent_p1;
        if (transform.kind == TransformKind::Flow) {
            NiftiImage temp(output, NiftiImage::Copy::ImageInfoAndAllocData);
            temp->intent_p1 = DEF_FIELD;
            reg_getDeformationFromDisplacement(temp);
            NiftiImage input_copy(transform.image, NiftiImage::Copy::Image);
            reg_getDisplacementFromDeformation(input_copy);
            reg_resampleGradient(input_copy, output, temp, 1, 0);
            reg_getDeformationFromDisplacement(output);
            output->intent_p2 *= -1.0f;
            result["kind"] = "flow";
        } else {
            NiftiImage temp(output, NiftiImage::Copy::ImageInfoAndAllocData);
            temp->intent_p1 = DEF_FIELD;
            reg_getDeformationFromDisplacement(temp);
            reg_resampleGradient(transform.image, output, temp, 1, 0);
            output->intent_p2 *= -1.0f;
            result["kind"] = "displacement_flow";
        }
        result["array"] = nifti_to_numpy(output);
        return result;
    }
    NiftiImage deformation = ensure_deformation_field(transform, reference_meta);
    NiftiImage inverse(deformation, NiftiImage::Copy::ImageInfoAndAllocData);
    set_vector_intent(inverse, DEF_FIELD);
    reg_defFieldInvert(deformation, inverse, 1.0e-6f);
    if (transform.kind == TransformKind::Displacement) {
        reg_getDisplacementFromDeformation(inverse);
        result["kind"] = "displacement";
    } else {
        result["kind"] = "deformation";
    }
    result["array"] = nifti_to_numpy(inverse);
    return result;
}

void take_log_in_place(NiftiImage &image) {
    float *ptr = static_cast<float *>(image->data);
    for (size_t i = 0; i < image->nvox; ++i) {
        ptr[i] = std::log(ptr[i]);
    }
}

NiftiImage create_scalar_output_like_reference(const ImageMeta &meta) {
    NiftiImage image(scalar_dims_from_meta(meta), DT_FLOAT32);
    apply_geometry(image, meta);
    image->scl_slope = 1.0f;
    image->scl_inter = 0.0f;
    return image;
}

NiftiImage create_jacobian_matrix_image(const ImageMeta &meta) {
    const int components = meta.shape_xyz.size() == 3 ? 9 : 4;
    NiftiImage image(vector_dims_from_meta(meta, components), DT_FLOAT32);
    apply_geometry(image, meta);
    image->intent_code = NIFTI_INTENT_VECTOR;
    return image;
}

py::dict jacobian_impl(
    const py::dict &reference_meta_dict,
    const py::dict &transform_payload,
    bool need_determinant,
    bool need_log_determinant,
    bool need_matrix
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    auto transform = parse_transform_payload(transform_payload);
    py::dict result;
    result["determinant"] = py::none();
    result["log_determinant"] = py::none();
    result["matrix"] = py::none();

    if (transform.kind == TransformKind::Cpp) {
        if (need_determinant || need_log_determinant) {
            NiftiImage jac = create_scalar_output_like_reference(reference_meta);
            reg_spline_GetJacobianMap(transform.image, jac);
            if (need_determinant) result["determinant"] = nifti_to_numpy(jac);
            if (need_log_determinant) {
                if (!need_determinant) reg_spline_GetJacobianMap(transform.image, jac);
                take_log_in_place(jac);
                result["log_determinant"] = nifti_to_numpy(jac);
            }
        }
        if (need_matrix) {
            NiftiImage matrix_image = create_jacobian_matrix_image(reference_meta);
            std::vector<mat33> matrices(reference_meta.shape_xyz[0] * reference_meta.shape_xyz[1] * (reference_meta.shape_xyz.size() == 3 ? reference_meta.shape_xyz[2] : 1));
            NiftiImage reference_image = create_scalar_output_like_reference(reference_meta);
            reg_spline_GetJacobianMatrix(reference_image, transform.image, matrices.data());
            float *ptr = static_cast<float *>(matrix_image->data);
            for (size_t voxel = 0; voxel < matrices.size(); ++voxel) {
                if (reference_meta.shape_xyz.size() == 3) {
                    for (int k = 0; k < 9; ++k) ptr[k * matrices.size() + voxel] = matrices[voxel].m[k / 3][k % 3];
                } else {
                    ptr[0 * matrices.size() + voxel] = matrices[voxel].m[0][0];
                    ptr[1 * matrices.size() + voxel] = matrices[voxel].m[0][1];
                    ptr[2 * matrices.size() + voxel] = matrices[voxel].m[1][0];
                    ptr[3 * matrices.size() + voxel] = matrices[voxel].m[1][1];
                }
            }
            result["matrix"] = nifti_to_numpy(matrix_image);
        }
        return result;
    }

    if (transform.kind == TransformKind::SplineVelocity) {
        if (need_determinant || need_log_determinant) {
            NiftiImage jac = create_scalar_output_like_reference(reference_meta);
            reg_spline_GetJacobianDetFromVelocityGrid(jac, transform.image);
            if (need_determinant) result["determinant"] = nifti_to_numpy(jac);
            if (need_log_determinant) {
                if (!need_determinant) reg_spline_GetJacobianDetFromVelocityGrid(jac, transform.image);
                take_log_in_place(jac);
                result["log_determinant"] = nifti_to_numpy(jac);
            }
        }
        if (need_matrix) {
            NiftiImage matrix_image = create_jacobian_matrix_image(reference_meta);
            std::vector<mat33> matrices(reference_meta.shape_xyz[0] * reference_meta.shape_xyz[1] * (reference_meta.shape_xyz.size() == 3 ? reference_meta.shape_xyz[2] : 1));
            NiftiImage reference_image = create_scalar_output_like_reference(reference_meta);
            reg_spline_GetJacobianMatFromVelocityGrid(matrices.data(), transform.image, reference_image);
            float *ptr = static_cast<float *>(matrix_image->data);
            for (size_t voxel = 0; voxel < matrices.size(); ++voxel) {
                if (reference_meta.shape_xyz.size() == 3) {
                    for (int k = 0; k < 9; ++k) ptr[k * matrices.size() + voxel] = matrices[voxel].m[k / 3][k % 3];
                } else {
                    ptr[0 * matrices.size() + voxel] = matrices[voxel].m[0][0];
                    ptr[1 * matrices.size() + voxel] = matrices[voxel].m[0][1];
                    ptr[2 * matrices.size() + voxel] = matrices[voxel].m[1][0];
                    ptr[3 * matrices.size() + voxel] = matrices[voxel].m[1][1];
                }
            }
            result["matrix"] = nifti_to_numpy(matrix_image);
        }
        return result;
    }

    if (transform.kind == TransformKind::Flow || transform.kind == TransformKind::DisplacementFlow) {
        NiftiImage flow = ensure_flow_field(transform, reference_meta);
        if (need_determinant || need_log_determinant) {
            NiftiImage jac = create_scalar_output_like_reference(reference_meta);
            reg_defField_GetJacobianDetFromFlowField(jac, flow);
            if (need_determinant) result["determinant"] = nifti_to_numpy(jac);
            if (need_log_determinant) {
                if (!need_determinant) reg_defField_GetJacobianDetFromFlowField(jac, flow);
                take_log_in_place(jac);
                result["log_determinant"] = nifti_to_numpy(jac);
            }
        }
        if (need_matrix) {
            NiftiImage matrix_image = create_jacobian_matrix_image(reference_meta);
            std::vector<mat33> matrices(reference_meta.shape_xyz[0] * reference_meta.shape_xyz[1] * (reference_meta.shape_xyz.size() == 3 ? reference_meta.shape_xyz[2] : 1));
            reg_defField_GetJacobianMatFromFlowField(matrices.data(), flow);
            float *ptr = static_cast<float *>(matrix_image->data);
            for (size_t voxel = 0; voxel < matrices.size(); ++voxel) {
                if (reference_meta.shape_xyz.size() == 3) {
                    for (int k = 0; k < 9; ++k) ptr[k * matrices.size() + voxel] = matrices[voxel].m[k / 3][k % 3];
                } else {
                    ptr[0 * matrices.size() + voxel] = matrices[voxel].m[0][0];
                    ptr[1 * matrices.size() + voxel] = matrices[voxel].m[0][1];
                    ptr[2 * matrices.size() + voxel] = matrices[voxel].m[1][0];
                    ptr[3 * matrices.size() + voxel] = matrices[voxel].m[1][1];
                }
            }
            result["matrix"] = nifti_to_numpy(matrix_image);
        }
        return result;
    }

    NiftiImage deformation = ensure_deformation_field(transform, reference_meta);
    if (need_determinant || need_log_determinant) {
        NiftiImage jac = create_scalar_output_like_reference(reference_meta);
        reg_defField_getJacobianMap(deformation, jac);
        if (need_determinant) result["determinant"] = nifti_to_numpy(jac);
        if (need_log_determinant) {
            if (!need_determinant) reg_defField_getJacobianMap(deformation, jac);
            take_log_in_place(jac);
            result["log_determinant"] = nifti_to_numpy(jac);
        }
    }
    if (need_matrix) {
        NiftiImage matrix_image = create_jacobian_matrix_image(reference_meta);
        std::vector<mat33> matrices(reference_meta.shape_xyz[0] * reference_meta.shape_xyz[1] * (reference_meta.shape_xyz.size() == 3 ? reference_meta.shape_xyz[2] : 1));
        reg_defField_getJacobianMatrix(deformation, matrices.data());
        float *ptr = static_cast<float *>(matrix_image->data);
        for (size_t voxel = 0; voxel < matrices.size(); ++voxel) {
            if (reference_meta.shape_xyz.size() == 3) {
                for (int k = 0; k < 9; ++k) ptr[k * matrices.size() + voxel] = matrices[voxel].m[k / 3][k % 3];
            } else {
                ptr[0 * matrices.size() + voxel] = matrices[voxel].m[0][0];
                ptr[1 * matrices.size() + voxel] = matrices[voxel].m[0][1];
                ptr[2 * matrices.size() + voxel] = matrices[voxel].m[1][0];
                ptr[3 * matrices.size() + voxel] = matrices[voxel].m[1][1];
            }
        }
        result["matrix"] = nifti_to_numpy(matrix_image);
    }
    return result;
}

std::unique_ptr<int[]> default_mask(size_t voxel_count) {
    auto mask = std::make_unique<int[]>(voxel_count);
    for (size_t i = 0; i < voxel_count; ++i) mask[i] = static_cast<int>(i);
    return mask;
}

py::dict measure_similarity_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &reference,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &floating,
    const py::dict &reference_meta_dict,
    const py::dict &floating_meta_dict,
    const py::object &reference_mask_obj,
    const py::object &transform_obj,
    int interpolation,
    float padding_value,
    const std::vector<std::string> &metrics
) {
    const ImageMeta reference_meta = parse_meta(reference_meta_dict);
    const ImageMeta floating_meta = parse_meta(floating_meta_dict);
    NiftiImage reference_image = make_scalar_image(reference, reference_meta, DT_FLOAT32);
    NiftiImage floating_image = make_scalar_image(floating, floating_meta, DT_FLOAT32);
    NiftiImage warped(reference_image, NiftiImage::Copy::ImageInfoAndAllocData);
    warped->datatype = floating_image->datatype;
    warped->nbyper = floating_image->nbyper;
    free(warped->data);
    warped->data = std::malloc(static_cast<size_t>(warped->nvox) * floating_image->nbyper);
    auto mask = reference_mask_obj.is_none()
        ? default_mask(reference_image.nVoxelsPerVolume())
        : make_mask_from_image(make_scalar_image(reference_mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(), reference_meta, DT_UINT8));
    NiftiImage deformation = transform_obj.is_none()
        ? create_identity_deformation(reference_meta)
        : ensure_deformation_field(parse_transform_payload(transform_obj.cast<py::dict>()), reference_meta);
    reg_resampleImage(floating_image, warped, deformation, mask.get(), interpolation, padding_value);
    py::dict out;
    for (const auto &metric : metrics) {
        if (metric == "ncc") {
            float *ref_ptr = static_cast<float *>(reference_image->data);
            float *war_ptr = static_cast<float *>(warped->data);
            double ref_mean = 0.0;
            double war_mean = 0.0;
            size_t count = 0;
            for (size_t i = 0; i < reference_image->nvox; ++i) {
                if (mask[i] > -1 && ref_ptr[i] == ref_ptr[i] && war_ptr[i] == war_ptr[i]) {
                    ref_mean += ref_ptr[i];
                    war_mean += war_ptr[i];
                    ++count;
                }
            }
            ref_mean /= static_cast<double>(count);
            war_mean /= static_cast<double>(count);
            double ref_std = 0.0;
            double war_std = 0.0;
            double corr = 0.0;
            for (size_t i = 0; i < reference_image->nvox; ++i) {
                if (mask[i] > -1 && ref_ptr[i] == ref_ptr[i] && war_ptr[i] == war_ptr[i]) {
                    ref_std += Square(static_cast<double>(ref_ptr[i]) - ref_mean);
                    war_std += Square(static_cast<double>(war_ptr[i]) - war_mean);
                    corr += (static_cast<double>(ref_ptr[i]) - ref_mean) * (static_cast<double>(war_ptr[i]) - war_mean);
                }
            }
            ref_std /= static_cast<double>(count);
            war_std /= static_cast<double>(count);
            out["ncc"] = corr / (std::sqrt(ref_std) * std::sqrt(war_std) * static_cast<double>(count));
        } else if (metric == "lncc") {
            reg_lncc lncc;
            lncc.SetTimePointWeight(0, 1.0);
            lncc.InitialiseMeasure(reference_image, warped, mask.get(), warped, nullptr, nullptr);
            out["lncc"] = lncc.GetSimilarityMeasureValue();
        } else if (metric == "nmi") {
            reg_nmi nmi;
            nmi.SetTimePointWeight(0, 1.0);
            nmi.InitialiseMeasure(reference_image, warped, mask.get(), warped, nullptr, nullptr);
            out["nmi"] = nmi.GetSimilarityMeasureValue();
        } else if (metric == "ssd") {
            reg_ssd ssd;
            ssd.SetTimePointWeight(0, 1.0);
            ssd.InitialiseMeasure(reference_image, warped, mask.get(), warped, nullptr, nullptr, nullptr);
            out["ssd"] = ssd.GetSimilarityMeasureValue();
        } else if (metric == "mind") {
            reg_mind mind;
            mind.SetTimePointWeight(0, 1.0);
            mind.InitialiseMeasure(reference_image, warped, mask.get(), warped, nullptr, nullptr);
            out["mind"] = mind.GetSimilarityMeasureValue();
        } else {
            throw std::invalid_argument("Unsupported metric: " + metric);
        }
    }
    return out;
}

mat44 average_affine_matrices(const std::vector<mat44> &input, float lts_inlier) {
    std::vector<mat44> matrices = input;
    for (auto &matrix : matrices) matrix = Mat44Logm(&matrix);
    std::vector<float> weights(matrices.size(), 1.0f);
    std::vector<int> sorted(matrices.size());
    std::iota(sorted.begin(), sorted.end(), 0);
    mat44 average {};
    const size_t iterations = lts_inlier < 1.0f && lts_inlier > 0.0f ? 10 : 1;
    for (size_t it = 0; it < iterations; ++it) {
        double sum[16] {};
        double weight_sum = 0.0;
        for (size_t i = 0; i < matrices.size(); ++i) {
            weight_sum += weights[i];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    sum[r * 4 + c] += matrices[i].m[r][c] * weights[i];
                }
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                average.m[r][c] = static_cast<float>(sum[r * 4 + c] / weight_sum);
            }
        }
        if (iterations > 1) {
            std::vector<float> distance(matrices.size(), 0.0f);
            for (size_t i = 0; i < matrices.size(); ++i) {
                mat44 minus = matrices[i] - average;
                mat44 minus_t;
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) minus_t.m[r][c] = minus.m[c][r];
                mat44 mtm = minus_t * minus;
                double trace = 0.0;
                for (int d = 0; d < 4; ++d) trace += mtm.m[d][d];
                if (trace < std::numeric_limits<double>::epsilon()) trace = std::numeric_limits<double>::epsilon();
                distance[i] = 1.0f / std::sqrt(trace);
                sorted[i] = static_cast<int>(i);
            }
            HeapSort(distance.data(), sorted.data(), distance.size());
            std::fill(weights.begin(), weights.end(), 0.0f);
            for (size_t m = distance.size() - 1; m > static_cast<size_t>(lts_inlier * distance.size()); --m) {
                weights[sorted[m]] = 1.0f;
            }
        }
        average = Mat44Expm(&average);
    }
    return average;
}

py::array_t<double> average_affines_impl(const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> &matrices, float lts_inlier) {
    std::vector<mat44> native;
    native.reserve(matrices.size());
    for (const auto &matrix : matrices) native.push_back(to_mat44(matrix));
    return mat44_to_numpy(average_affine_matrices(native, lts_inlier));
}

void write_debug_image_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &image,
    const py::dict &meta_dict,
    const std::string &path
) {
    const ImageMeta meta = parse_meta(meta_dict);
    NiftiImage native = make_scalar_image(image, meta, DT_FLOAT32);
    reg_io_WriteImageFile(native, path.c_str());
}

}  // namespace

PYBIND11_MODULE(_niftyreg, module) {
    module.def("run_aladin", &run_aladin_impl);
    module.def("run_aladin_from_files", &run_aladin_from_files_impl);
    module.def("run_f3d", &run_f3d_impl);
    module.def("resample_image", &resample_impl);
    module.def("transform_to_deformation", &transform_to_deformation_impl);
    module.def("transform_to_displacement", &transform_to_displacement_impl);
    module.def("transform_to_flow", &transform_to_flow_impl);
    module.def("compose_transforms", &compose_transforms_impl);
    module.def("invert_transform", &invert_transform_impl);
    module.def("jacobian", &jacobian_impl);
    module.def("measure_similarity", &measure_similarity_impl);
    module.def("average_affines", &average_affines_impl);
    module.def("write_debug_image", &write_debug_image_impl);
}
