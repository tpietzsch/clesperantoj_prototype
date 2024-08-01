
package net.clesperanto.kernels;

import java.util.Objects;
import java.util.ArrayList;

import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.Utils;

/**
 * Class containing all functions of tier 1 category
 */
public class Tier1 {

	/**
	 * Computes the absolute value of every individual pixel x in a given image.
	 * <pre>f(x) = |x| </pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_absolute
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ absolute(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.absolute(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Calculates the sum of pairs of pixels x and y from images X and Y weighted with factors a and b.
	 * <pre>f(x, y, a, b) = x * a + y * b</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first input image to added.
	 * @param input1 ({@link ArrayJ}) - The second image to be added.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @param factor0 (float) - Multiplication factor of each pixel of src0 before adding it. (default: 1)
	 * @param factor1 (float) - Multiplication factor of each pixel of src1 before adding it. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_addImagesWeighted
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ addImagesWeighted(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output, float factor0, float factor1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.add_images_weighted(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw(), factor0, factor1), device);
    }
    
	/**
	 * Adds a scalar value s to all pixels x of a given image X.
	 * <pre>f(x, s) = x + s</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image where scalare should be added.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @param scalar (float) - The constant number which will be added to all pixels. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_addImageAndScalar
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ addImageAndScalar(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.add_image_and_scalar(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Computes a binary image (containing pixel values 0 and 1) from two images X and Y by connecting pairs of pixels x and y with the binary AND operator &.
	 * All pixel values except 0 in the input images are interpreted as 1.
	 * <pre>f(x, y) = x & y</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first binary input image to be processed.
	 * @param input1 ({@link ArrayJ}) - The second binary input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binaryAnd
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryAnd(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_and(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines pixels/voxels which are on the surface of binary objects and sets only them to 1 in the destination image.
	 * All other pixels are set to 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The binary input image where edges will be searched.
	 * @param output ({@link ArrayJ}) - The output image where edge pixels will be 1. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binaryEdgeDetection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryEdgeDetection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_edge_detection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image (containing pixel values 0 and 1) from an image X by negating its pixel values x using the binary NOT operator ! All pixel values except 0 in the input image are interpreted as 1.
	 * <pre>f(x) = !x</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The binary input image to be inverted.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binaryNot
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryNot(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_not(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image (containing pixel values 0 and 1) from two images X and Y by connecting pairs of pixels x and y with the binary OR operator |.
	 * All pixel values except 0 in the input images are interpreted as 1.
	 * <pre>f(x, y) = x | y</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first binary input image to be processed.
	 * @param input1 ({@link ArrayJ}) - The second binary input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binaryOr
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryOr(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_or(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Subtracts one binary image from another.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first binary input image to be processed.
	 * @param input1 ({@link ArrayJ}) - The second binary input image to be subtracted from the first.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binarySubtract
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binarySubtract(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_subtract(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image (containing pixel values 0 and 1) from two images X and Y by connecting pairs of pixels x and y with the binary operators AND &, OR | and NOT ! implementing the XOR operator.
	 * All pixel values except 0 in the input images are interpreted as 1.
	 * <pre>f(x, y) = (x & !y) | (!x & y)</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first binary input image to be processed.
	 * @param input1 ({@link ArrayJ}) - The second binary input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_binaryXOr
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryXor(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_xor(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Compute the maximum of the erosion with plannar structuring elements.
	 * Warning: This operation is only supported BINARY data type images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The binary input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binarySupinf(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_supinf(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Compute the minimum of the dilation with plannar structuring elements.
	 * Warning: This operation is only supported BINARY data type images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The binary input image to be processed.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ binaryInfsup(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.binary_infsup(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Enumerates pixels with value 1 in a onedimensional image For example handing over the image [0, 1, 1, 0, 1, 0, 1, 1] would be processed to an image [0, 1, 2, 0, 3, 0, 4, 5] This functionality is important in connected component neccessary (see also sum_reduction_x).
	 * In the above example, with blocksize 4, that would be the sum array: [2, 3] labeling.
	 * Processing is accelerated by paralellization in blocks.
	 * Therefore, handing over precomputed block sums is Note that the block size when calling this function and sum_reduction must be identical.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - input binary vector image
	 * @param input1 ({@link ArrayJ}) - precomputed sums of blocks
	 * @param output ({@link ArrayJ}) - output enumerated vector image (default: None)
	 * @param blocksize (int) -  (default: 256)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ blockEnumerate(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output, int blocksize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.block_enumerate(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw(), blocksize), device);
    }
    
	/**
	 * Convolve the image with a given kernel image.
	 * It is recommended that the kernel image has an odd size in X, Y and Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_convolve
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ convolve(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.convolve(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Copies an image.
	 * <pre>f(x) = x</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to copy.
	 * @param output ({@link ArrayJ}) - Output copy image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_copy
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ copy(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.copy(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * This method has two purposes: It copies a 2D image to a given slice z position in a 3D image stack or It copies a given slice at position z in an image stack to a 2D image.
	 * The first case is only available via ImageJ macro.
	 * If you are using it, it is recommended that the target 3D image already preexists in GPU memory before calling this method.
	 * Otherwise, CLIJ create the image stack with z planes.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to copy from.
	 * @param output ({@link ArrayJ}) - Output copy image slice. (default: None)
	 * @param slice (int) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_copySlice
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ copySlice(DeviceJ device, ArrayJ input, ArrayJ output, int slice) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.copy_slice(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), slice), device);
    }
    
	/**
	 * This method has two purposes: It copies a 2D image to a given slice y position in a 3D image stack or It copies a given slice at position y in an image stack to a 2D image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to copy from.
	 * @param output ({@link ArrayJ}) - Output copy image slice. (default: None)
	 * @param slice (int) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_copySlice
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ copyHorizontalSlice(DeviceJ device, ArrayJ input, ArrayJ output, int slice) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.copy_horizontal_slice(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), slice), device);
    }
    
	/**
	 * This method has two purposes: It copies a 2D image to a given slice x position in a 3D image stack or It copies a given slice at position x in an image stack to a 2D image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to copy from.
	 * @param output ({@link ArrayJ}) - Output copy image slice. (default: None)
	 * @param slice (int) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_copySlice
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ copyVerticalSlice(DeviceJ device, ArrayJ input, ArrayJ output, int slice) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.copy_vertical_slice(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), slice), device);
    }
    
	/**
	 * Crops a given substack out of a given image stack.
	 * Note: If the destination image preexists already, it will be overwritten and keep it's dimensions.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param start_x (int) - Starting index coordicante x. (default: 0)
	 * @param start_y (int) - Starting index coordicante y. (default: 0)
	 * @param start_z (int) - Starting index coordicante z. (default: 0)
	 * @param width (int) - Width size of the region to crop. (default: 1)
	 * @param height (int) - Height size of the region to crop. (default: 1)
	 * @param depth (int) - Depth size of the region to crop. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_crop3D
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ crop(DeviceJ device, ArrayJ input, ArrayJ output, int start_x, int start_y, int start_z, int width, int height, int depth) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.crop(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), start_x, start_y, start_z, width, height, depth), device);
    }
    
	/**
	 * Computes the cubic root of each pixel.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ cubicRoot(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.cubic_root(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Takes a labelmap and returns an image where all pixels on label edges are set to 1 and all other pixels to 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_detectLabelEdges
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ detectLabelEdges(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.detect_label_edges(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary dilation of a given input image.
	 * The dilation takes the Mooreneighborhood (8 pixels in 2D and 26 pixels in 3d) into account.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * This method is comparable to the 'Dilate' menu in ImageJ in case it is applied to a 2D image.
	 * The only difference is that the output image contains values 0 and 1 instead of 0 and 255.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process. Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_dilateBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ dilateBox(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.dilate_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary dilation of a given input image.
	 * The dilation takes the vonNeumannneighborhood (4 pixels in 2D and 6 pixels in 3d) into account.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process. Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_dilateSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ dilateSphere(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.dilate_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary dilation of a given input image.
	 * The dilation apply the Mooreneighborhood (8 pixels in 2D and 26 pixels in 3d) for the "box" connectivity and the vonNeumannneighborhood (4 pixels in 2D and 6 pixels in 3d) for a "sphere" connectivity.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process. Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. Output result image. (default: None)
	 * @param connectivity (String) - Element shape, "box" or "sphere". (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_dilateBox
	 * @see https://clij.github.io/clij2-docs/reference_dilateSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ dilate(DeviceJ device, ArrayJ input, ArrayJ output, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.dilate(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), connectivity), device);
    }
    
	/**
	 * Divides two images X and Y by each other pixel wise.
	 * <pre>f(x, y) = x / y</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_divideImages
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ divideImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.divide_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Divides a scalar by an image pixel by pixel.
	 * <pre>f(x, s) = s / x</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ divideScalarByImage(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.divide_scalar_by_image(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Determines if two images A and B equal pixel wise.
	 * <pre>f(a, b) = 1 if a == b; 0 otherwise.
	 * </pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first image to be compared with.
	 * @param input1 ({@link ArrayJ}) - The second image to be compared with the first.
	 * @param output ({@link ArrayJ}) - The resulting binary image where pixels will be 1 only if source1 (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_equal
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ equal(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.equal(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if an image A and a constant b are equal.
	 * <pre>f(a, b) = 1 if a == b; 0 otherwise.
	 * </pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The image where every pixel is compared to the constant.
	 * @param output ({@link ArrayJ}) - The resulting binary image where pixels will be 1 only if source1 (default: None)
	 * @param scalar (float) - The constant where every pixel is compared to. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_equalConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ equalConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.equal_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary erosion of a given input image.
	 * The erosion takes the Mooreneighborhood (8 pixels in 2D and 26 pixels in 3d) into account.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * This method is comparable to the 'Erode' menu in ImageJ in case it is applied to a 2D image.
	 * The only difference is that the output image contains values 0 and 1 instead of 0 and 255.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_erodeBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ erodeBox(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.erode_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary erosion of a given input image.
	 * The erosion takes the vonNeumannneighborhood (4 pixels in 2D and 6 pixels in 3d) into account.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_erodeSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ erodeSphere(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.erode_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a binary image with pixel values 0 and 1 containing the binary erosion of a given input image.
	 * The erosion apply the Mooreneighborhood (8 pixels in 2D and 26 pixels in 3d) for the "box" connectivity and the vonNeumannneighborhood (4 pixels in 2D and 6 pixels in 3d) for a "sphere" connectivity.
	 * The pixels in the input image with pixel value not equal to 0 will be interpreted as 1.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param connectivity (String) - Element shape, "box" or "sphere". (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_erodeBox
	 * @see https://clij.github.io/clij2-docs/reference_erodeSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ erode(DeviceJ device, ArrayJ input, ArrayJ output, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.erode(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), connectivity), device);
    }
    
	/**
	 * Computes base exponential of all pixels values.
	 * f(x) = exp(x) Author(s): Peter Haub, Robert Haase.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_exponential
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ exponential(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.exponential(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Flips an image in X, Y and/or Z direction depending on boolean flags.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param flip_x (boolean) - Flip along the x axis if true. (default: True)
	 * @param flip_y (boolean) - Flip along the y axis if true. (default: True)
	 * @param flip_z (boolean) - Flip along the z axis if true. (default: True)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_flip3D
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ flip(DeviceJ device, ArrayJ input, ArrayJ output, boolean flip_x, boolean flip_y, boolean flip_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.flip(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), flip_x, flip_y, flip_z), device);
    }
    
	/**
	 * Computes the Gaussian blurred image of an image given sigma values in X, Y and Z.
	 * Thus, the filter kernel can have nonisotropic shape.
	 * The implementation is done separable.
	 * In case a sigma equals zero, the direction is not blurred.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param sigma_x (float) - Sigma value along the x axis. (default: 0)
	 * @param sigma_y (float) - Sigma value along the y axis. (default: 0)
	 * @param sigma_z (float) - Sigma value along the z axis. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_gaussianBlur3D
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ gaussianBlur(DeviceJ device, ArrayJ input, ArrayJ output, float sigma_x, float sigma_y, float sigma_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.gaussian_blur(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), sigma_x, sigma_y, sigma_z), device);
    }
    
	/**
	 * Computes the distance between all point coordinates given in two point lists.
	 * Takes two images containing pointlists (dimensionality n * d, n: number of points and d: dimensionality) and builds up a matrix containing the distances between these points.
	 * Convention: Given two point lists with dimensionality n * d and m * d, the distance matrix will be of size(n + 1) * (m + 1).
	 * The first row and column contain zeros.
	 * They represent the distance of the (see generateTouchMatrix).
	 * Thus, one can threshold a distance matrix to generate a touch matrix out of it for drawing objects to a theoretical background object.
	 * In that way, distance matrices are of the same size as touch matrices meshes.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_generateDistanceMatrix
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ generateDistanceMatrix(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.generate_distance_matrix(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the gradient of gray values along X.
	 * Assuming a, b and c are three adjacent pixels in X direction.
	 * In the target image will be saved as: <pre>b' = c a;</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_gradientX
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ gradientX(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.gradient_x(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the gradient of gray values along Y.
	 * Assuming a, b and c are three adjacent pixels in Y direction.
	 * In the target image will be saved as: <pre>b' = c a;</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_gradientY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ gradientY(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.gradient_y(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the gradient of gray values along Z.
	 * Assuming a, b and c are three adjacent pixels in Z direction.
	 * In the target image will be saved as: <pre>b' = c a;</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_gradientZ
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ gradientZ(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.gradient_z(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B greater pixel wise.
	 * f(a, b) = 1 if a > b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_greater
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ greater(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.greater(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B greater pixel wise.
	 * f(a, b) = 1 if a > b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_greaterConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ greaterConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.greater_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Determines if two images A and B greater or equal pixel wise.
	 * f(a, b) = 1 if a >= b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_greaterOrEqual
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ greaterOrEqual(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.greater_or_equal(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B greater or equal pixel wise.
	 * f(a, b) = 1 if a >= b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar value used in the comparison. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_greaterOrEqualConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ greaterOrEqualConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.greater_or_equal_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Computes the eigenvalues of the hessian matrix of a 2d or 3d image.
	 * Hessian matrix or 2D images: [Ixx, Ixy] [Ixy, Iyy] Hessian matrix for 3D images: [Ixx, Ixy, Ixz] [Ixy, Iyy, Iyz] [Ixz, Iyz, Izz] Ixx denotes the second derivative in x.
	 * Ixx and Iyy are calculated by convolving the image with the 1d kernel [1 2 1].
	 * Ixy is calculated by a convolution with the 2d kernel: [ 0.
	 * 25 0 0.
	 * 25] [ 0 0 0] [0.
	 * 25 0 0.
	 * 25] Note: This is the only clesperanto function that returns multiple images.
	 * This API might be subject to change in the future.
	 * Consider using small_hessian_eigenvalue() and/or large_hessian_eigenvalue() instead which return only one image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param small_eigenvalue ({@link ArrayJ}) - Output result image. (default: None)
	 * @param middle_eigenvalue ({@link ArrayJ}) - Output result image, null if input is 2D. (default: None)
	 * @param large_eigenvalue ({@link ArrayJ}) - Output result image. (default: None)
	 * @return ArrayList<{@link ArrayJ}>
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayList<ArrayJ> hessianEigenvalues(DeviceJ device, ArrayJ input, ArrayJ small_eigenvalue, ArrayJ middle_eigenvalue, ArrayJ large_eigenvalue) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return Utils.toArrayList(net.clesperanto._internals.kernelj.Tier1.hessian_eigenvalues(device.getRaw(), input.getRaw(), small_eigenvalue == null ? null : small_eigenvalue.getRaw(), middle_eigenvalue == null ? null : middle_eigenvalue.getRaw(), large_eigenvalue == null ? null : large_eigenvalue.getRaw()));
    }
    
	/**
	 * Applies the Laplace operator (Box neighborhood) to an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_laplaceBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ laplaceBox(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.laplace_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Applies the Laplace operator (Diamond neighborhood) to an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_laplaceDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ laplaceDiamond(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.laplace_diamond(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Applies the Laplace operator with a "box" or a "sphere" neighborhood to an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_laplaceDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ laplace(DeviceJ device, ArrayJ input, ArrayJ output, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.laplace(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), connectivity), device);
    }
    
	/**
	 * Compute the cross correlation of an image to a given kernel.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ localCrossCorrelation(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.local_cross_correlation(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes base e logarithm of all pixels values.
	 * f(x) = log(x) Author(s): Peter Haub, Robert Haase.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_logarithm
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ logarithm(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.logarithm(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a masked image by applying a binary mask to an image.
	 * All pixel values x of image X will be copied to the destination image in case pixel value m at the same position in the mask image is not equal to zero.
	 * <pre>f(x,m) = (x if (m != 0); (0 otherwise))</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param mask ({@link ArrayJ}) - Mask image to apply.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_mask
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ mask(DeviceJ device, ArrayJ input, ArrayJ mask, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
		Objects.requireNonNull(mask, "mask cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mask(device.getRaw(), input.getRaw(), mask.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes a masked image by applying a label mask to an image.
	 * All pixel values x of image X will be copied to the destination image in case pixel value m at the same position in the label_map image has the right index value i.
	 * f(x,m,i) = (x if (m == i); (0 otherwise)).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - Input Intensity image.
	 * @param input1 ({@link ArrayJ}) - Input Label image.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param label (float) - Label value to use. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maskLabel
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maskLabel(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output, float label) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mask_label(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw(), label), device);
    }
    
	/**
	 * Computes the maximum of a constant scalar s and each pixel value x in a given image X.
	 * <pre>f(x, s) = max(x, s)</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar value used in the comparison. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximumImageAndScalar
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumImageAndScalar(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_image_and_scalar(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Computes the maximum of a pair of pixel values x, y from two given images X and Y.
	 * <pre>f(x, y) = max(x, y)</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximumImages
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the local maximum of a pixels cube neighborhood.
	 * The cubes size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximum3DBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local maximum of a pixels neighborhood (box or sphere).
	 * The neighborhood size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 0)
	 * @param radius_y (int) - Radius size along y axis. (default: 0)
	 * @param radius_z (int) - Radius size along z axis. (default: 0)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximum3DBox
	 * @see https://clij.github.io/clij2-docs/reference_maximum3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximum(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Determines the maximum intensity projection of an image along X.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximumXProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the maximum intensity projection of an image along X.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximumYProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the maximum intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximumZProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the local mean average of a pixels boxshaped neighborhood.
	 * The cubes size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_mean3DBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ meanBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local mean average of a pixels spherical neighborhood.
	 * The spheres size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_mean3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ meanSphere(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local mean average of a pixels neighborhood defined as a boxshaped or a sphereshaped.
	 * The shape size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_mean3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ mean(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Determines the mean average intensity projection of an image along X.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_meanXProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ meanXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the mean average intensity projection of an image along Y.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_meanYProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ meanYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the mean average intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_meanZProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ meanZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mean_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the local median of a pixels box shaped neighborhood.
	 * The box is specified by its halfwidth and halfheight (radius).
	 * For technical reasons, the area of the box must have less than 1000 pixels.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_median3DBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ medianBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.median_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local median of a pixels sphere shaped neighborhood.
	 * The sphere is specified by its halfwidth and halfheight (radius).
	 * For technical reasons, the area of the box must have less than 1000 pixels.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_median3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ medianSphere(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.median_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local median of a pixels neighborhood.
	 * The neighborhood is defined as a box or a sphere shape.
	 * Its size is specified by its halfwidth, halfheight, and halfdepth (radius).
	 * For technical reasons, the area of the shpae must have less than 1000 pixels.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_median3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ median(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.median(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Computes the local minimum of a pixels cube neighborhood.
	 * The cubes size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 0)
	 * @param radius_y (int) - Radius size along y axis. (default: 0)
	 * @param radius_z (int) - Radius size along z axis. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimum3DBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local minimum of a pixels cube neighborhood.
	 * The cubes size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 0)
	 * @param radius_y (int) - Radius size along y axis. (default: 0)
	 * @param radius_z (int) - Radius size along z axis. (default: 0)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimum3DBox
	 * @see https://clij.github.io/clij2-docs/reference_minimum3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimum(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Computes the minimum of a constant scalar s and each pixel value x in a given image X.
	 * <pre>f(x, s) = min(x, s)</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar value used in the comparison. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimumImageAndScalar
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumImageAndScalar(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_image_and_scalar(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Computes the minimum of a pair of pixel values x, y from two given images X and Y.
	 * <pre>f(x, y) = min(x, y)</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimumImages
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the minimum intensity projection of an image along Y.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimumXProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the minimum intensity projection of an image along Y.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimumYProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the minimum intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimumZProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the local mode of a pixels box shaped neighborhood.
	 * This can be used to postprocess and locally correct semantic segmentation results.
	 * The box is specified by its halfwidth and halfheight (radius).
	 * For technical reasons, the intensities must lie within a range from 0 to 255.
	 * In case multiple values have maximum frequency, the smallest one is returned.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ modeBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mode_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local mode of a pixels sphere shaped neighborhood.
	 * This can be used to postprocess and locally correct semantic segmentation results.
	 * The sphere is specified by its halfwidth and halfheight (radius).
	 * For technical reasons, the intensities must lie within a range from 0 to 255.
	 * In case multiple values have maximum frequency, the smallest one is returned.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ modeSphere(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mode_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local mode of a pixels neighborhood.
	 * This neighborhood can be shaped as a box or a sphere.
	 * This can be used to postprocess and locally correct semantic segmentation results.
	 * The shape size is specified by its halfwidth, halfheight, and halfdepth (radius).
	 * For technical reasons, the intensities must lie within a range from 0 to 255 (uint8).
	 * In case multiple values have maximum frequency, the smallest one is returned.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ mode(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.mode(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Computes the remainder of a division of pairwise pixel values in two images.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ moduloImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.modulo_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Multiplies all pixel intensities with the x, y or z coordinate, depending on specified dimension.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param dimension (int) - Dimension (0,1,2) to use in the operation. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_multiplyImageAndCoordinate
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ multiplyImageAndPosition(DeviceJ device, ArrayJ input, ArrayJ output, int dimension) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.multiply_image_and_position(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), dimension), device);
    }
    
	/**
	 * Multiplies all pixels value x in a given image X with a constant scalar s.
	 * <pre>f(x, s) = x * s</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image to be multiplied with a constant.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @param scalar (float) - The number with which every pixel will be multiplied with. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_multiplyImageAndScalar
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ multiplyImageAndScalar(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.multiply_image_and_scalar(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Multiplies all pairs of pixel values x and y from two image X and Y.
	 * <pre>f(x, y) = x * y</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first input image to be multiplied.
	 * @param input1 ({@link ArrayJ}) - The second image to be multiplied.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_multiplyImages
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ multiplyImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.multiply_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Copies all pixels instead those which are not a number (NaN), or positive/negative infinity which are replaced by a defined new value, default 0.
	 * This function aims to work similarly as its counterpart in numpy [1].
	 * Default values for posinf and neginf may differ from numpy and even differ depending on compute hardware.
	 * It is recommended to specify those values.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @param nan (float) - Value to replace (default: 0)
	 * @param posinf (float) - Value to replace +inf with. (default: np.nan_to_num(float('inf')))
	 * @param neginf (float) - Value to replace -inf with. (default: np.nan_to_num(float('-inf')))
	 * @return {@link ArrayJ}
	 * @see https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nanToNum(DeviceJ device, ArrayJ input, ArrayJ output, float nan, float posinf, float neginf) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nan_to_num(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), nan, posinf, neginf), device);
    }
    
	/**
	 * Apply a maximum filter (box shape) to the input image.
	 * The radius is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMaximumBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMaximumBox(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_maximum_box(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a maximum filter (diamond shape) to the input image.
	 * The radius is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMaximumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMaximumDiamond(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_maximum_diamond(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a maximum filter of a neighborhood to the input image.
	 * The neighborhood shape can be a box or a sphere.
	 * The size is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMaximumBox
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMaximumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMaximum(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_maximum(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw(), connectivity), device);
    }
    
	/**
	 * Apply a minimum filter (box shape) to the input image.
	 * The radius is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMinimumBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMinimumBox(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_minimum_box(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a minimum filter (diamond shape) to the input image.
	 * The radius is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMinimumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMinimumDiamond(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_minimum_diamond(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a minimum filter of a neighborhood to the input image.
	 * The neighborhood shape can be a box or a sphere.
	 * The radius is fixed to 1 and pixels with value 0 are ignored.
	 * Note: Pixels with 0 value in the input image will not be overwritten in the output image.
	 * Thus, the result image should be initialized by copying the original image in advance.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag (0 or 1).
	 * @param output1 ({@link ArrayJ}) - Output image where results are written into. (default: None)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMinimumBox
	 * @see https://clij.github.io/clij2-docs/reference_nonzeroMinimumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ nonzeroMinimum(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.nonzero_minimum(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw(), connectivity), device);
    }
    
	/**
	 * Determines if two images A and B equal pixel wise.
	 * f(a, b) = 1 if a != b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - The first image to be compared with.
	 * @param input1 ({@link ArrayJ}) - The second image to be compared with the first.
	 * @param output ({@link ArrayJ}) - The resulting binary image where pixels will be 1 only if source1 (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_notEqual
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ notEqual(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.not_equal(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B equal pixel wise.
	 * f(a, b) = 1 if a != b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The image where every pixel is compared to the constant.
	 * @param output ({@link ArrayJ}) - The resulting binary image where pixels will be 1 only if source1 (default: None)
	 * @param scalar (float) - The constant where every pixel is compared to. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_notEqualConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ notEqualConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.not_equal_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Pastes an image into another image at a given position.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param index_x (int) - Origin pixel coodinate in x to paste. (default: 0)
	 * @param index_y (int) - Origin pixel coodinate in y to paste. (default: 0)
	 * @param index_z (int) - Origin pixel coodinate in z to paste. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_paste3D
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ paste(DeviceJ device, ArrayJ input, ArrayJ output, int index_x, int index_y, int index_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.paste(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), index_x, index_y, index_z), device);
    }
    
	/**
	 * Apply a local maximum filter to an image which only overwrites pixels with value 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag value, 0 or 1.
	 * @param output1 ({@link ArrayJ}) - Output image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_onlyzeroOverwriteMaximumBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ onlyzeroOverwriteMaximumBox(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.onlyzero_overwrite_maximum_box(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a local maximum filter to an image which only overwrites pixels with value 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag value, 0 or 1.
	 * @param output1 ({@link ArrayJ}) - Output image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_onlyzeroOverwriteMaximumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ onlyzeroOverwriteMaximumDiamond(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.onlyzero_overwrite_maximum_diamond(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw()), device);
    }
    
	/**
	 * Apply a local maximum filter to an image which only overwrites pixels with value 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output0 ({@link ArrayJ}) - Output flag value, 0 or 1.
	 * @param output1 ({@link ArrayJ}) - Output image. (default: None)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_onlyzeroOverwriteMaximumBox
	 * @see https://clij.github.io/clij2-docs/reference_onlyzeroOverwriteMaximumDiamond
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ onlyzeroOverwriteMaximum(DeviceJ device, ArrayJ input, ArrayJ output0, ArrayJ output1, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.onlyzero_overwrite_maximum(device.getRaw(), input.getRaw(), output0.getRaw(), output1 == null ? null : output1.getRaw(), connectivity), device);
    }
    
	/**
	 * Computes all pixels value x to the power of a given exponent a.
	 * <pre>f(x, a) = x ^ a</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Power value. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_power
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ power(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.power(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Calculates x to the power of y pixel wise of two images X and Y.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_powerImages
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ powerImages(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.power_images(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Crops an image according to a defined range and step size.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - First input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param start_x (int) - Range starting value in x (default: None)
	 * @param stop_x (int) - Range stop value in x (default: None)
	 * @param step_x (int) - Range step value in x (default: None)
	 * @param start_y (int) - Range starting value in y (default: None)
	 * @param stop_y (int) - Range stop value in y (default: None)
	 * @param step_y (int) - Range step value in y (default: None)
	 * @param start_z (int) - Range starting value in z (default: None)
	 * @param stop_z (int) - Range stop value in z (default: None)
	 * @param step_z (int) - Range step value in z (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ range(DeviceJ device, ArrayJ input, ArrayJ output, int start_x, int stop_x, int step_x, int start_y, int stop_y, int step_y, int start_z, int stop_z, int step_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.range(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), start_x, stop_x, step_x, start_y, stop_y, step_y, start_z, stop_z, step_z), device);
    }
    
	/**
	 * Go to positions in a given image specified by a pointlist and read intensities of those pixels.
	 * The intensities are stored in a new vector.
	 * The positions are passed as a (x,y,z) coordinate per column.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param list ({@link ArrayJ}) - List of coordinate, as a 2D matrix.
	 * @param output ({@link ArrayJ}) - Output vector image of intensities. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ readValuesFromPositions(DeviceJ device, ArrayJ input, ArrayJ list, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
		Objects.requireNonNull(list, "list cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.read_values_from_positions(device.getRaw(), input.getRaw(), list.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Replaces integer intensities specified in a vector image.
	 * The values are passed as a vector of values.
	 * The vector index represents the old intensity and the value at that position represents the new intensity.
	 * s.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - Input image to process.
	 * @param input1 ({@link ArrayJ}) - List of intensities to replace, as a vector of values.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_replaceIntensities
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ replaceValues(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.replace_values(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Replaces a specific intensity in an image with a given new value.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar0 (float) - Old value. (default: 0)
	 * @param scalar1 (float) - New value. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_replaceIntensity
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ replaceValue(DeviceJ device, ArrayJ input, ArrayJ output, float scalar0, float scalar1) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.replace_value(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar0, scalar1), device);
    }
    
	/**
	 * Computes the local maximum of a pixels spherical neighborhood.
	 * The spheres size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (float) - Radius size along x axis. (default: 1)
	 * @param radius_y (float) - Radius size along y axis. (default: 1)
	 * @param radius_z (float) - Radius size along z axis. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_maximum3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ maximumSphere(DeviceJ device, ArrayJ input, ArrayJ output, float radius_x, float radius_y, float radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.maximum_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local minimum of a pixels spherical neighborhood.
	 * The spheres size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (float) - Radius size along x axis. (default: 1)
	 * @param radius_y (float) - Radius size along y axis. (default: 1)
	 * @param radius_z (float) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_minimum3DSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ minimumSphere(DeviceJ device, ArrayJ input, ArrayJ output, float radius_x, float radius_y, float radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.minimum_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Multiplies two matrices with each other.
	 * Shape of matrix1 should be equal to shape of matrix2 transposed.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_multiplyMatrix
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ multiplyMatrix(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.multiply_matrix(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes 1/x for every pixel value This function is supposed to work similarly to its counter part in numpy [1].
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ reciprocal(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.reciprocal(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Sets all pixel values x of a given image X to a constant value v.
	 * <pre>f(x) = v</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param scalar (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_set
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ set(DeviceJ device, ArrayJ input, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set(device.getRaw(), input.getRaw(), scalar), device);
    }
    
	/**
	 * Sets all pixel values x of a given column in X to a constant value v.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param column (int) - Column index. (default: 0)
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setColumn
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setColumn(DeviceJ device, ArrayJ input, int column, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_column(device.getRaw(), input.getRaw(), column, value), device);
    }
    
	/**
	 * Sets all pixel values at the image border to a given value.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setImageBorders
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setImageBorders(DeviceJ device, ArrayJ input, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_image_borders(device.getRaw(), input.getRaw(), value), device);
    }
    
	/**
	 * Sets all pixel values x of a given plane in X to a constant value v.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param plane (int) - Plane index. (default: 0)
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setPlane
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setPlane(DeviceJ device, ArrayJ input, int plane, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_plane(device.getRaw(), input.getRaw(), plane, value), device);
    }
    
	/**
	 * Sets all pixel values to their X coordinate.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setRampX
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setRampX(DeviceJ device, ArrayJ input) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_ramp_x(device.getRaw(), input.getRaw()), device);
    }
    
	/**
	 * Sets all pixel values to their Y coordinate.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setRampY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setRampY(DeviceJ device, ArrayJ input) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_ramp_y(device.getRaw(), input.getRaw()), device);
    }
    
	/**
	 * Sets all pixel values to their Z coordinate.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setRampZ
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setRampZ(DeviceJ device, ArrayJ input) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_ramp_z(device.getRaw(), input.getRaw()), device);
    }
    
	/**
	 * Sets all pixel values x of a given row in X to a constant value v.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param row (int) -  (default: 0)
	 * @param value (float) -  (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setRow
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setRow(DeviceJ device, ArrayJ input, int row, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_row(device.getRaw(), input.getRaw(), row, value), device);
    }
    
	/**
	 * Replaces all 0 value pixels in an image with the index of a pixel.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output image. (default: None)
	 * @param offset (int) - Offset value to start the indexing. (default: 1)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setNonzeroPixelsToPixelindex(DeviceJ device, ArrayJ input, ArrayJ output, int offset) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_nonzero_pixels_to_pixelindex(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), offset), device);
    }
    
	/**
	 * Sets all pixel values a of a given image A to a constant value v in case its coordinates x == y.
	 * Otherwise the pixel is not overwritten.
	 * If you want to initialize an identity transfrom matrix, set all pixels to 0 first.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setWhereXequalsY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setWhereXEqualsY(DeviceJ device, ArrayJ input, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_where_x_equals_y(device.getRaw(), input.getRaw(), value), device);
    }
    
	/**
	 * Sets all pixel values a of a given image A to a constant value v in case its coordinates x > y.
	 * Otherwise the pixel is not overwritten.
	 * If you want to initialize an identity transfrom matrix, set all pixels to 0 first.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setWhereXgreaterThanY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setWhereXGreaterThanY(DeviceJ device, ArrayJ input, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_where_x_greater_than_y(device.getRaw(), input.getRaw(), value), device);
    }
    
	/**
	 * Sets all pixel values a of a given image A to a constant value v in case its coordinates x < y.
	 * Otherwise the pixel is not overwritten.
	 * If you want to initialize an identity transfrom matrix, set all pixels to 0 first.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param value (float) - Value to set. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_setWhereXsmallerThanY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ setWhereXSmallerThanY(DeviceJ device, ArrayJ input, float value) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.set_where_x_smaller_than_y(device.getRaw(), input.getRaw(), value), device);
    }
    
	/**
	 * Extracts the sign of pixels.
	 * If a pixel value < 0, resulting pixel value will be 1.
	 * If it was > 0, it will be 1.
	 * Otherwise it will be 0.
	 * This function aims to work similarly as its counterpart in numpy [1].
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sign(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sign(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B smaller pixel wise.
	 * f(a, b) = 1 if a < b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_smaller
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smaller(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.smaller(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B smaller pixel wise.
	 * f(a, b) = 1 if a < b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar used in the comparison. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_smallerConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smallerConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.smaller_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Determines if two images A and B smaller or equal pixel wise.
	 * f(a, b) = 1 if a <= b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input0 ({@link ArrayJ}) - First input image to process.
	 * @param input1 ({@link ArrayJ}) - Second input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_smallerOrEqual
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smallerOrEqual(DeviceJ device, ArrayJ input0, ArrayJ input1, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input0, "input0 cannot be null");
		Objects.requireNonNull(input1, "input1 cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.smaller_or_equal(device.getRaw(), input0.getRaw(), input1.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines if two images A and B smaller or equal pixel wise.
	 * f(a, b) = 1 if a <= b; 0 otherwise.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar used in the comparison. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_smallerOrEqualConstant
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ smallerOrEqualConstant(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.smaller_or_equal_constant(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Convolve the image with the Sobel kernel.
	 * Author(s): Ruth WhelanJeans, Robert Haase.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_sobel
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sobel(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sobel(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the square root of each pixel.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ squareRoot(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.square_root(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the standard deviation intensity projection of an image stack along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_standardDeviationZProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ stdZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.std_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Subtracts one image X from a scalar s pixel wise.
	 * <pre>f(x, s) = s x</pre>.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param scalar (float) - Scalar used in the subtraction. (default: 0)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_subtractImageFromScalar
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ subtractImageFromScalar(DeviceJ device, ArrayJ input, ArrayJ output, float scalar) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.subtract_image_from_scalar(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), scalar), device);
    }
    
	/**
	 * Takes an image and reduces it in width by factor blocksize.
	 * The new pixels contain the sum of the reduced pixels.
	 * For example, given the following image and block size 4: [0, 1, 1, 0, 1, 0, 1, 1] would lead to an image [2, 3].
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param blocksize (int) - Blocksize value. (default: 256)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sumReductionX(DeviceJ device, ArrayJ input, ArrayJ output, int blocksize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sum_reduction_x(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), blocksize), device);
    }
    
	/**
	 * Determines the sum intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_sumXProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sumXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sum_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the sum intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_sumYProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sumYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sum_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines the sum intensity projection of an image along Z.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_sumZProjection
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ sumZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.sum_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Transpose X and Y axes of an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_transposeXY
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ transposeXy(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.transpose_xy(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Transpose X and Z axes of an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_transposeXZ
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ transposeXz(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.transpose_xz(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Transpose Y and Z axes of an image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - The input image.
	 * @param output ({@link ArrayJ}) - The output image where results are written into. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_transposeYZ
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ transposeYz(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.transpose_yz(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Copies all pixels instead those which are not a number (NaN) or infinity (inf), which are replaced by 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_undefinedToZero
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ undefinedToZero(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.undefined_to_zero(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Computes the local variance of a pixels box neighborhood.
	 * The box size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * If 2D images are given, radius_z will be ignored.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_varianceBox
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ varianceBox(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.variance_box(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local variance of a pixels sphere neighborhood.
	 * The sphere size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * If 2D images are given, radius_z will be ignored.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_varianceSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ varianceSphere(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.variance_sphere(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z), device);
    }
    
	/**
	 * Computes the local variance of a pixels neighborhood (box or sphere).
	 * The neighborhood size is specified by its halfwidth, halfheight and halfdepth (radius).
	 * If 2D images are given, radius_z will be ignored.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @param radius_x (int) - Radius size along x axis. (default: 1)
	 * @param radius_y (int) - Radius size along y axis. (default: 1)
	 * @param radius_z (int) - Radius size along z axis. (default: 1)
	 * @param connectivity (String) - Filter neigborhood (default: "box")
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_varianceBox
	 * @see https://clij.github.io/clij2-docs/reference_varianceSphere
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ variance(DeviceJ device, ArrayJ input, ArrayJ output, int radius_x, int radius_y, int radius_z, String connectivity) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.variance(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius_x, radius_y, radius_z, connectivity), device);
    }
    
	/**
	 * Takes an image with three/four rows (2D: height = 3; 3D: height = 4): x, y [, z] and v and target image.
	 * The value v will be written at position x/y[/z] in the target image.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image to process.
	 * @param output ({@link ArrayJ}) - Output result image. (default: None)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_writeValuesToPositions
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ writeValuesToPositions(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.write_values_to_positions(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines an Xposition of the maximum intensity along X and writes it into the resulting image.
	 * If there are multiple xslices with the same value, the smallest X will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ xPositionOfMaximumXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.x_position_of_maximum_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines an Xposition of the minimum intensity along X and writes it into the resulting image.
	 * If there are multiple xslices with the same value, the smallest X will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ xPositionOfMinimumXProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.x_position_of_minimum_x_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines an Yposition of the maximum intensity along Y and writes it into the resulting image.
	 * If there are multiple yslices with the same value, the smallest Y will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ yPositionOfMaximumYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.y_position_of_maximum_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines an Yposition of the minimum intensity along Y and writes it into the resulting image.
	 * If there are multiple yslices with the same value, the smallest Y will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ yPositionOfMinimumYProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.y_position_of_minimum_y_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines a Zposition of the maximum intensity along Z and writes it into the resulting image.
	 * If there are multiple zslices with the same value, the smallest Z will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ zPositionOfMaximumZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.z_position_of_maximum_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
	/**
	 * Determines a Zposition of the minimum intensity along Z and writes it into the resulting image.
	 * If there are multiple zslices with the same value, the smallest Z will be chosen.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input image stack
	 * @param output ({@link ArrayJ}) - altitude map (default: None)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ zPositionOfMinimumZProjection(DeviceJ device, ArrayJ input, ArrayJ output) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier1.z_position_of_minimum_z_projection(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw()), device);
    }
    
}
