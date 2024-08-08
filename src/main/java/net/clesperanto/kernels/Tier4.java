
package net.clesperanto.kernels;

import java.util.Objects;
import java.util.ArrayList;
import java.util.HashMap;

import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.Utils;

/**
 * Class containing all functions of tier 4 category
 */
public class Tier4 {

	/**
	 * Determines the bounding box of the specified label from a label image.
	 * The positions are returned in  an array of 6 values as follows: minX, minY, minZ, maxX, maxY, maxZ.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - 
	 * @param label_id (int) - 
	 * @return ArrayList<Float>
	 * @see https://clij.github.io/clij2-docs/reference_boundingBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayList<Float> labelBoundingBox(DeviceJ device, ArrayJ input, int label_id) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return Utils.toArrayList(net.clesperanto._internals.kernelj.Tier4.label_bounding_box(device.getRaw(), input.getRaw(), label_id));
    }
    
	/**
	 * Determines the mean squared error (MSE) between two images.
	 * The MSE will be stored in a new row of ImageJs Results table in the column 'MSE'.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - 
	 * @param input1 ({@link ArrayJ}) - 
	 * @return float
	 * @see https://clij.github.io/clij2-docs/reference_meanSquaredError
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static float meanSquaredError(DeviceJ device, ArrayJ input0, ArrayJ input1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return net.clesperanto._internals.kernelj.Tier4.mean_squared_error(device.getRaw(), input0.getRaw(), input1.getRaw());
    }
    
	/**
	 * Transforms a spots image as resulting from maximum/minimum detection in an image where every column contains d pixels (with d = dimensionality of the original image) with the coordinates of the maxima/minima.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - 
	 * @param output ({@link ArrayJ}) -  (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_spotsToPointList
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ spotsToPointlist(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier4.spots_to_pointlist(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Analyses a label map and if there are gaps in the indexing (e.
	 * g.
	 * label 5 is not present) all subsequent labels will be relabelled.
	 * Thus, afterwards number of labels and maximum label index are equal.
	 * This operation is mostly performed on the CPU.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - 
	 * @param output ({@link ArrayJ}) -  (default: None)
	 * @param blocksize (int) - Renumbering is done in blocks for performance reasons. (default: 4096)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_closeIndexGapsInLabelMap
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ relabelSequential(DeviceJ device, ArrayJ input, ArrayJ output, int blocksize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier4.relabel_sequential(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), blocksize), device);
    }
    
	/**
	 * Binarizes an image using Otsu's threshold method [3] implemented in scikit-image[2] using a histogram determined on the GPU to create binary images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - 
	 * @param output ({@link ArrayJ}) -  (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_thresholdOtsu
	 * @see https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu
	 * @see https://ieeexplore.ieee.org/document/4310076
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ thresholdOtsu(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier4.threshold_otsu(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
}
