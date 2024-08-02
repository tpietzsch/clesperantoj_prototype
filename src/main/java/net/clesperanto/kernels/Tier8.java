
package net.clesperanto.kernels;

import java.util.Objects;
import java.util.ArrayList;
import java.util.HashMap;

import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.Utils;

/**
 * Class containing all functions of tier 8 category
 */
public class Tier8 {

	/**
	 * Apply a morphological opening operation to a label image and afterwards   fills gaps between the labels using voronoi-labeling.
	 * Finally, the result   label image is masked so that all background pixels remain background pixels.
	 * Note: It is recommended to process isotropic label images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input label image
	 * @param output ({@link ArrayJ}) - Output label image (default: None)
	 * @param radius (int) - Smoothing (default: 0)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smoothLabels(DeviceJ device, ArrayJ input, ArrayJ output, int radius) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier8.smooth_labels(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius), device);
    }

	/**
	 * Apply a morphological erosion and dilation of the label image with respect to     the connectivity of the labels.
	 * Note: It is recommended to process isotropic label images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input label image
	 * @param output ({@link ArrayJ}) - Output label image (default: None)
	 * @param radius (int) - Smoothing (default: 0)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smoothConnectedLabels(DeviceJ device, ArrayJ input, ArrayJ output, int radius) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier8.smooth_connected_labels(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius), device);
    }

}
