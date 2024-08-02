
#include "kernelj.hpp"
#include "tier5.hpp"

bool Tier5::array_equal(DeviceJ * device, ArrayJ * src0, ArrayJ * src1)
{
    return cle::tier5::array_equal_func(device->get(), src0->get(), src1->get());
}

ArrayJ Tier5::combine_labels(DeviceJ * device, ArrayJ * src0, ArrayJ * src1, ArrayJ * dst)
{
    return ArrayJ{cle::tier5::combine_labels_func(device->get(), src0->get(), src1->get(), dst == nullptr ? nullptr : dst->get())};
}

ArrayJ Tier5::connected_components_labeling(DeviceJ * device, ArrayJ * src, ArrayJ * dst, std::string connectivity)
{
    return ArrayJ{cle::tier5::connected_components_labeling_func(device->get(), src->get(), dst == nullptr ? nullptr : dst->get(), connectivity)};
}

