
#include "kernelj.hpp"
#include "tier1.hpp"

ArrayJ Tier1::absolute(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::absolute_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::add_images_weighted(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst, float factor0, float factor1)
{
    return ArrayJ{cle::tier1::add_images_weighted_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get(), factor0, factor1)};
}

ArrayJ Tier1::add_image_and_scalar(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::add_image_and_scalar_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::binary_and(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_and_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_edge_detection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_edge_detection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_not(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_not_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_or(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_or_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_subtract(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_subtract_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_xor(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_xor_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_supinf(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_supinf_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::binary_infsup(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::binary_infsup_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::block_enumerate(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst, int blocksize)
{
    return ArrayJ{cle::tier1::block_enumerate_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get(), blocksize)};
}

ArrayJ Tier1::convolve(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::convolve_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::copy(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::copy_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::copy_slice(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int slice)
{
    return ArrayJ{cle::tier1::copy_slice_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), slice)};
}

ArrayJ Tier1::copy_horizontal_slice(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int slice)
{
    return ArrayJ{cle::tier1::copy_horizontal_slice_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), slice)};
}

ArrayJ Tier1::copy_vertical_slice(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int slice)
{
    return ArrayJ{cle::tier1::copy_vertical_slice_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), slice)};
}

ArrayJ Tier1::crop(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int start_x, int start_y, int start_z, int width, int height, int depth)
{
    return ArrayJ{cle::tier1::crop_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), start_x, start_y, start_z, width, height, depth)};
}

ArrayJ Tier1::cubic_root(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::cubic_root_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::detect_label_edges(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::detect_label_edges_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::dilate_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::dilate_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::dilate_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::dilate_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::dilate(DeviceJ * device, ArrayJ * src, ArrayJ * dst, std::string connectivity)
{
    return ArrayJ{cle::tier1::dilate_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), connectivity)};
}

ArrayJ Tier1::divide_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::divide_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::divide_scalar_by_image(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::divide_scalar_by_image_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::equal(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::equal_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::equal_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::equal_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::erode_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::erode_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::erode_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::erode_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::erode(DeviceJ * device, ArrayJ * src, ArrayJ * dst, std::string connectivity)
{
    return ArrayJ{cle::tier1::erode_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), connectivity)};
}

ArrayJ Tier1::exponential(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::exponential_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::flip(DeviceJ * device, ArrayJ * src, ArrayJ * dst, bool flip_x, bool flip_y, bool flip_z)
{
    return ArrayJ{cle::tier1::flip_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), flip_x, flip_y, flip_z)};
}

ArrayJ Tier1::gaussian_blur(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float sigma_x, float sigma_y, float sigma_z)
{
    return ArrayJ{cle::tier1::gaussian_blur_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), sigma_x, sigma_y, sigma_z)};
}

ArrayJ Tier1::generate_distance_matrix(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::generate_distance_matrix_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::gradient_x(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::gradient_x_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::gradient_y(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::gradient_y_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::gradient_z(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::gradient_z_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::greater(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::greater_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::greater_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::greater_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::greater_or_equal(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::greater_or_equal_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::greater_or_equal_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::greater_or_equal_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

std::vector<ArrayJ> Tier1::hessian_eigenvalues(DeviceJ * device, ArrayJ * src, ArrayJ * small_eigenvalue, ArrayJ * middle_eigenvalue, ArrayJ * large_eigenvalue)
{
    return UtilsJ::toArrayJVector(cle::tier1::hessian_eigenvalues_func(device->get(), src->get(), small_eigenvalue == nullptr ? nullptr : small_eigenvalue->get(), middle_eigenvalue == nullptr ? nullptr : middle_eigenvalue->get(), large_eigenvalue == nullptr ? nullptr : large_eigenvalue->get()));
}

ArrayJ Tier1::laplace_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::laplace_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::laplace_diamond(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::laplace_diamond_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::laplace(DeviceJ * device, ArrayJ * src, ArrayJ * dst, std::string connectivity)
{
    return ArrayJ{cle::tier1::laplace_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), connectivity)};
}

ArrayJ Tier1::local_cross_correlation(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::local_cross_correlation_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::logarithm(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::logarithm_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mask(DeviceJ * device, ArrayJ * src, ArrayJ * mask, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::mask_func(device->get(), src->get(), mask->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mask_label(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst, float label)
{
    return ArrayJ{cle::tier1::mask_label_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get(), label)};
}

ArrayJ Tier1::maximum_image_and_scalar(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::maximum_image_and_scalar_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::maximum_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::maximum_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::maximum_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::maximum_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::maximum(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::maximum_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::maximum_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::maximum_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::maximum_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::maximum_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::maximum_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::maximum_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mean_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::mean_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::mean_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::mean_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::mean(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::mean_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::mean_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::mean_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mean_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::mean_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mean_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::mean_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::median_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::median_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::median_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::median_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::median(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::median_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::minimum_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::minimum_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::minimum(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::minimum_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::minimum_image_and_scalar(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::minimum_image_and_scalar_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::minimum_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::minimum_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::minimum_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::minimum_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::minimum_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::minimum_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::minimum_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::minimum_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::mode_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::mode_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::mode_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::mode_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::mode(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::mode_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::modulo_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::modulo_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::multiply_image_and_position(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int dimension)
{
    return ArrayJ{cle::tier1::multiply_image_and_position_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), dimension)};
}

ArrayJ Tier1::multiply_image_and_scalar(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::multiply_image_and_scalar_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::multiply_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::multiply_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::nan_to_num(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float nan, float posinf, float neginf)
{
    return ArrayJ{cle::tier1::nan_to_num_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), nan, posinf, neginf)};
}

ArrayJ Tier1::nonzero_maximum_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::nonzero_maximum_box_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::nonzero_maximum_diamond(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::nonzero_maximum_diamond_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::nonzero_maximum(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1, std::string connectivity)
{
    return ArrayJ{cle::tier1::nonzero_maximum_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get(), connectivity)};
}

ArrayJ Tier1::nonzero_minimum_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::nonzero_minimum_box_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::nonzero_minimum_diamond(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::nonzero_minimum_diamond_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::nonzero_minimum(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1, std::string connectivity)
{
    return ArrayJ{cle::tier1::nonzero_minimum_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get(), connectivity)};
}

ArrayJ Tier1::not_equal(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::not_equal_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::not_equal_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::not_equal_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::paste(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int index_x, int index_y, int index_z)
{
    return ArrayJ{cle::tier1::paste_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), index_x, index_y, index_z)};
}

ArrayJ Tier1::onlyzero_overwrite_maximum_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::onlyzero_overwrite_maximum_box_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::onlyzero_overwrite_maximum_diamond(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1)
{
    return ArrayJ{cle::tier1::onlyzero_overwrite_maximum_diamond_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get())};
}

ArrayJ Tier1::onlyzero_overwrite_maximum(DeviceJ * device, ArrayJ * src, ArrayJ * dst0, ArrayJ * dst1, std::string connectivity)
{
    return ArrayJ{cle::tier1::onlyzero_overwrite_maximum_func(device->get(), src->get(), dst0->get(), dst1 == nullptr ? nullptr : dst1->get(), connectivity)};
}

ArrayJ Tier1::power(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::power_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::power_images(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::power_images_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::range(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int start_x, int stop_x, int step_x, int start_y, int stop_y, int step_y, int start_z, int stop_z, int step_z)
{
    return ArrayJ{cle::tier1::range_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), start_x, stop_x, step_x, start_y, stop_y, step_y, start_z, stop_z, step_z)};
}

ArrayJ Tier1::read_values_from_positions(DeviceJ * device, ArrayJ * src, ArrayJ * list, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::read_values_from_positions_func(device->get(), src->get(), list->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::replace_values(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::replace_values_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::replace_value(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar0, float scalar1)
{
    return ArrayJ{cle::tier1::replace_value_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar0, scalar1)};
}

ArrayJ Tier1::maximum_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float radius_x, float radius_y, float radius_z)
{
    return ArrayJ{cle::tier1::maximum_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::minimum_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float radius_x, float radius_y, float radius_z)
{
    return ArrayJ{cle::tier1::minimum_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::multiply_matrix(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::multiply_matrix_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::reciprocal(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::reciprocal_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::set(DeviceJ * device, ArrayJ * src, float scalar)
{
    return ArrayJ{cle::tier1::set_func(device->get(), src->get(), scalar)};
}

ArrayJ Tier1::set_column(DeviceJ * device, ArrayJ * src, int column, float value)
{
    return ArrayJ{cle::tier1::set_column_func(device->get(), src->get(), column, value)};
}

ArrayJ Tier1::set_image_borders(DeviceJ * device, ArrayJ * src, float value)
{
    return ArrayJ{cle::tier1::set_image_borders_func(device->get(), src->get(), value)};
}

ArrayJ Tier1::set_plane(DeviceJ * device, ArrayJ * src, int plane, float value)
{
    return ArrayJ{cle::tier1::set_plane_func(device->get(), src->get(), plane, value)};
}

ArrayJ Tier1::set_ramp_x(DeviceJ * device, ArrayJ * src)
{
    return ArrayJ{cle::tier1::set_ramp_x_func(device->get(), src->get())};
}

ArrayJ Tier1::set_ramp_y(DeviceJ * device, ArrayJ * src)
{
    return ArrayJ{cle::tier1::set_ramp_y_func(device->get(), src->get())};
}

ArrayJ Tier1::set_ramp_z(DeviceJ * device, ArrayJ * src)
{
    return ArrayJ{cle::tier1::set_ramp_z_func(device->get(), src->get())};
}

ArrayJ Tier1::set_row(DeviceJ * device, ArrayJ * src, int row, float value)
{
    return ArrayJ{cle::tier1::set_row_func(device->get(), src->get(), row, value)};
}

ArrayJ Tier1::set_nonzero_pixels_to_pixelindex(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int offset)
{
    return ArrayJ{cle::tier1::set_nonzero_pixels_to_pixelindex_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), offset)};
}

ArrayJ Tier1::set_where_x_equals_y(DeviceJ * device, ArrayJ * src, float value)
{
    return ArrayJ{cle::tier1::set_where_x_equals_y_func(device->get(), src->get(), value)};
}

ArrayJ Tier1::set_where_x_greater_than_y(DeviceJ * device, ArrayJ * src, float value)
{
    return ArrayJ{cle::tier1::set_where_x_greater_than_y_func(device->get(), src->get(), value)};
}

ArrayJ Tier1::set_where_x_smaller_than_y(DeviceJ * device, ArrayJ * src, float value)
{
    return ArrayJ{cle::tier1::set_where_x_smaller_than_y_func(device->get(), src->get(), value)};
}

ArrayJ Tier1::sign(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::sign_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::smaller(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::smaller_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::smaller_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::smaller_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::smaller_or_equal(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::smaller_or_equal_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::smaller_or_equal_constant(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::smaller_or_equal_constant_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::sobel(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::sobel_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::square_root(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::square_root_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::std_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::std_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::subtract_image_from_scalar(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float scalar)
{
    return ArrayJ{cle::tier1::subtract_image_from_scalar_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), scalar)};
}

ArrayJ Tier1::sum_reduction_x(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int blocksize)
{
    return ArrayJ{cle::tier1::sum_reduction_x_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), blocksize)};
}

ArrayJ Tier1::sum_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::sum_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::sum_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::sum_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::sum_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::sum_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::transpose_xy(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::transpose_xy_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::transpose_xz(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::transpose_xz_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::transpose_yz(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::transpose_yz_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::undefined_to_zero(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::undefined_to_zero_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::variance_box(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::variance_box_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::variance_sphere(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z)
{
    return ArrayJ{cle::tier1::variance_sphere_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z)};
}

ArrayJ Tier1::variance(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int radius_x, int radius_y, int radius_z, std::string connectivity)
{
    return ArrayJ{cle::tier1::variance_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), radius_x, radius_y, radius_z, connectivity)};
}

ArrayJ Tier1::write_values_to_positions(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::write_values_to_positions_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::x_position_of_maximum_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::x_position_of_maximum_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::x_position_of_minimum_x_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::x_position_of_minimum_x_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::y_position_of_maximum_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::y_position_of_maximum_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::y_position_of_minimum_y_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::y_position_of_minimum_y_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::z_position_of_maximum_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::z_position_of_maximum_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier1::z_position_of_minimum_z_projection(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier1::z_position_of_minimum_z_projection_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}
