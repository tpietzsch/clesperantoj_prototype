
#include "kernelj.hpp"
#include "tier3.hpp"

std::vector<float> Tier3::bounding_box(DeviceJ * device, ArrayJ * src)
{
    return cle::tier3::bounding_box_func(device->get(), src->get());
}

std::vector<float> Tier3::center_of_mass(DeviceJ * device, ArrayJ * src)
{
    return cle::tier3::center_of_mass_func(device->get(), src->get());
}

ArrayJ Tier3::exclude_labels(DeviceJ * device, ArrayJ * src, ArrayJ * list, ArrayJ * dst)
{
    return ArrayJ{cle::tier3::exclude_labels_func(device->get(), src->get(), list->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier3::exclude_labels_on_edges(DeviceJ * device, ArrayJ * src, ArrayJ * dst, bool exclude_x, bool exclude_y, bool exclude_z)
{
    return ArrayJ{cle::tier3::exclude_labels_on_edges_func(device == nullptr ? nullptr : device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), exclude_x, exclude_y, exclude_z)};
}

ArrayJ Tier3::flag_existing_labels(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier3::flag_existing_labels_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier3::gamma_correction(DeviceJ * device, ArrayJ * src, ArrayJ * dst, float gamma)
{
    return ArrayJ{cle::tier3::gamma_correction_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), gamma)};
}

ArrayJ Tier3::generate_binary_overlap_matrix(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier3::generate_binary_overlap_matrix_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier3::generate_touch_matrix(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier3::generate_touch_matrix_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier3::histogram(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int nbins, float min, float max)
{
    return ArrayJ{cle::tier3::histogram_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), nbins, min, max)};
}

float Tier3::jaccard_index(DeviceJ * device, ArrayJ * src0, ArrayJ * src1)
{
    return cle::tier3::jaccard_index_func(device->get(), src0->get(), src1->get());
}

ArrayJ Tier3::labelled_spots_to_pointlist(DeviceJ * device, ArrayJ * src, ArrayJ * dst)
{
    return ArrayJ{cle::tier3::labelled_spots_to_pointlist_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get())};
}

std::vector<float> Tier3::maximum_position(DeviceJ * device, ArrayJ * src)
{
    return cle::tier3::maximum_position_func(device->get(), src->get());
}

float Tier3::mean_of_all_pixels(DeviceJ * device, ArrayJ * src)
{
    return cle::tier3::mean_of_all_pixels_func(device->get(), src->get());
}

std::vector<float> Tier3::minimum_position(DeviceJ * device, ArrayJ * src)
{
    return cle::tier3::minimum_position_func(device->get(), src->get());
}

ArrayJ Tier3::morphological_chan_vese(DeviceJ * device, ArrayJ * src, ArrayJ * dst, int num_iter, int smoothing, float lambda1, float lambda2)
{
    return ArrayJ{cle::tier3::morphological_chan_vese_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), num_iter, smoothing, lambda1, lambda2)};
}

std::unordered_map<std::string, std::vector<float>> Tier3::statistics_of_labelled_pixels(DeviceJ * device, ArrayJ * src, ArrayJ * intensity)
{
    return cle::tier3::statistics_of_labelled_pixels_func(device->get(), src->get(), intensity == nullptr ? nullptr : intensity->get());
}
