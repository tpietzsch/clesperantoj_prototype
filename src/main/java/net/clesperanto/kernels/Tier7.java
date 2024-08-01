
package net.clesperanto.kernels;

import java.util.Objects;
import java.util.ArrayList;

import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.Utils;

/**
 * Class containing all functions of tier 7 category
 */
public class Tier7 {

	/**
	 * Apply an affine transformation matrix to an array and return the result.
	 * The transformation matrix must be 3x3 or 4x4 stored as a 1D array.
	 * The matrix should be row-major, i.
	 * e.
	 * the first 3 elements are the first row of the matrix.
	 * If no matrix is given, the identity matrix will be used.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be transformed.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param transform_matrix (ArrayList<Float>) - Affine transformation matrix (3x3 or 4x4). (default: None)
	 * @param interpolate (boolean) - If true, bi/trilinear interpolation will be applied, if hardware allows. (default: False)
	 * @param resize (boolean) - Automatically determines the size of the output depending on the rotation angles. (default: False)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ affineTransform(DeviceJ device, ArrayJ input, ArrayJ output, ArrayList<Float> transform_matrix, boolean interpolate, boolean resize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.affine_transform(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), Utils.toVector(transform_matrix), interpolate, resize), device);
    }
    
	/**
	 * Segments and labels an image using blurring, Otsu-thresholding, binary erosion and  masked Voronoi-labeling.
	 * After bluring and Otsu-thresholding the image, iterative binary erosion is applied.
	 * Objects in the eroded image are labeled and the labels are extended to fit again into  the initial binary image using masked-Voronoi labeling.
	 * This function is similar to voronoi_otsu_labeling.
	 * It is intended to deal better in  case labels of objects swapping into each other if objects are dense.
	 * Like when using  Voronoi-Otsu-labeling, small objects may disappear when applying this operation.
	 * This function is inspired by a similar implementation in Java by Jan Brocher (Biovoxxel) [0] [1].
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be transformed.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param number_of_erosions (int) - Number of iteration of erosion. (default: 5)
	 * @param outline_sigma (float) - Gaussian blur sigma applied before Otsu thresholding. (default: 2)
	 * @return {@link ArrayJ}
	 * @see https://github.com/biovoxxel/bv3dbox (BV_LabelSplitter.java#L83)
	 * @see https://zenodo.org/badge/latestdoi/434949702
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ erodedOtsuLabeling(DeviceJ device, ArrayJ input, ArrayJ output, int number_of_erosions, float outline_sigma) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.eroded_otsu_labeling(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), number_of_erosions, outline_sigma), device);
    }
    
	/**
	 * Translate the image by a given vector and rotate it by given angles.
	 * Angles are given in degrees.
	 * To convert radians to degrees, use this formula: angle_in_degrees = angle_in_radians / numpy.
	 * pi * 180.
	 * 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be transformed.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param translate_x (float) - Translation along x axis in pixels. (default: 0)
	 * @param translate_y (float) - Translation along y axis in pixels. (default: 0)
	 * @param translate_z (float) - Translation along z axis in pixels. (default: 0)
	 * @param angle_x (float) - Rotation around x axis in radians. (default: 0)
	 * @param angle_y (float) - Rotation around y axis in radians. (default: 0)
	 * @param angle_z (float) - Rotation around z axis in radians. (default: 0)
	 * @param centered (boolean) - If true, rotate image around center, else around the origin. (default: True)
	 * @param interpolate (boolean) - If true, bi/trilinear interpolation will be applied, if hardware allows. (default: False)
	 * @param resize (boolean) - Automatically determines the size of the output depending on the rotation angles. (default: False)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ rigidTransform(DeviceJ device, ArrayJ input, ArrayJ output, float translate_x, float translate_y, float translate_z, float angle_x, float angle_y, float angle_z, boolean centered, boolean interpolate, boolean resize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.rigid_transform(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), translate_x, translate_y, translate_z, angle_x, angle_y, angle_z, centered, interpolate, resize), device);
    }
    
	/**
	 * Rotate the image by given angles.
	 * Angles are given in degrees.
	 * To convert radians to degrees, use this formula: angle_in_degrees = angle_in_radians / numpy.
	 * pi * 180.
	 * 0.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be rotated.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param angle_x (float) - Rotation around x axis in degrees. (default: 0)
	 * @param angle_y (float) - Rotation around y axis in degrees. (default: 0)
	 * @param angle_z (float) - Rotation around z axis in degrees. (default: 0)
	 * @param centered (boolean) - If true, rotate image around center, else around the origin. (default: True)
	 * @param interpolate (boolean) - If true, bi/trilinear interpolation will be applied, if hardware allows. (default: False)
	 * @param resize (boolean) - Automatically determines the size of the output depending on the rotation angles. (default: False)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ rotate(DeviceJ device, ArrayJ input, ArrayJ output, float angle_x, float angle_y, float angle_z, boolean centered, boolean interpolate, boolean resize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.rotate(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), angle_x, angle_y, angle_z, centered, interpolate, resize), device);
    }
    
	/**
	 * Scale the image by given factors.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be scaleded.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param factor_x (float) - Scaling along x axis. (default: 1)
	 * @param factor_y (float) - Scaling along y axis. (default: 1)
	 * @param factor_z (float) - Scaling along z axis. (default: 1)
	 * @param centered (boolean) - If true, the image will be scaled to the center of the image. (default: True)
	 * @param interpolate (boolean) - If true, bi/trilinear interplation will be applied. (default: False)
	 * @param resize (boolean) - Automatically determines output size image. (default: False)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ scale(DeviceJ device, ArrayJ input, ArrayJ output, float factor_x, float factor_y, float factor_z, boolean centered, boolean interpolate, boolean resize) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.scale(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), factor_x, factor_y, factor_z, centered, interpolate, resize), device);
    }
    
	/**
	 * Translate the image by a given vector.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input Array to be translated.
	 * @param output ({@link ArrayJ}) - Output Array. (default: None)
	 * @param translate_x (float) - Translation along x axis in pixels. (default: 0)
	 * @param translate_y (float) - Translation along y axis in pixels. (default: 0)
	 * @param translate_z (float) - Translation along z axis in pixels. (default: 0)
	 * @param interpolate (boolean) - If true, bi/trilinear interplation will be applied. (default: False)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ translate(DeviceJ device, ArrayJ input, ArrayJ output, float translate_x, float translate_y, float translate_z, boolean interpolate) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.translate(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), translate_x, translate_y, translate_z, interpolate), device);
    }
    
	/**
	 * Apply a morphological closing operation to a label image.
	 * The operation consists of iterative dilation and erosion of the labels.
	 * With every iteration, box and diamond/sphere structuring elements are used and thus, the operation has an octagon as structuring element.
	 * Notes * This operation assumes input images are isotropic.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input label Array.
	 * @param output ({@link ArrayJ}) - Output label Array. (default: None)
	 * @param radius (int) - Radius size for the closing. (default: 0)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ closingLabels(DeviceJ device, ArrayJ input, ArrayJ output, int radius) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.closing_labels(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius), device);
    }
    
	/**
	 * Erodes labels to a smaller size.
	 * Note: Depending on the label image and the radius,  labels may disappear and labels may split into multiple islands.
	 * Thus, overlapping labels of input and output may  not have the same identifier.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - result
	 * @param output ({@link ArrayJ}) -  (default: None)
	 * @param radius (int) -  (default: 1)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ erodeConnectedLabels(DeviceJ device, ArrayJ input, ArrayJ output, int radius) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.erode_connected_labels(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius), device);
    }
    
	/**
	 * Apply a morphological opening operation to a label image.
	 * The operation consists of iterative erosion and dilation of the labels.
	 * With every iteration, box and diamond/sphere structuring elements are used and thus, the operation has an octagon as structuring element.
	 * Notes * This operation assumes input images are isotropic.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input label Array.
	 * @param output ({@link ArrayJ}) - Output label Array. (default: None)
	 * @param radius (int) - Radius size for the opening. (default: 0)
	 * @return {@link ArrayJ}
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ openingLabels(DeviceJ device, ArrayJ input, ArrayJ output, int radius) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.opening_labels(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), radius), device);
    }
    
	/**
	 * Labels objects directly from greyvalue images.
	 * The two sigma parameters allow tuning the segmentation result.
	 * Under the hood, this filter applies two Gaussian blurs, spot detection, Otsuthresholding [2] and Voronoilabeling [3].
	 * The thresholded binary image is flooded using the Voronoi tesselation approach starting from the found local maxima.
	 * Notes * This operation assumes input images are isotropic.
	 * @param device ({@link DeviceJ}) - Device to perform the operation on.
	 * @param input ({@link ArrayJ}) - Input intensity Array.
	 * @param output ({@link ArrayJ}) - Output label Array. (default: None)
	 * @param spot_sigma (float) - Controls how close detected cells can be. (default: 2)
	 * @param outline_sigma (float) - Controls how precise segmented objects are outlined. (default: 2)
	 * @return {@link ArrayJ}
	 * @see https://clij.github.io/clij2-docs/reference_voronoiOtsuLabeling
	 * @see https://ieeexplore.ieee.org/document/4310076
	 * @see https://en.wikipedia.org/wiki/Voronoi_diagram
	 * @throws NullPointerException if any of the device or input parameters are null.
	 */
    public static ArrayJ voronoiOtsuLabeling(DeviceJ device, ArrayJ input, ArrayJ output, float spot_sigma, float outline_sigma) {
        Objects.requireNonNull(device, "device cannot be null");
		Objects.requireNonNull(input, "input cannot be null");
        return new ArrayJ(net.clesperanto._internals.kernelj.Tier7.voronoi_otsu_labeling(device.getRaw(), input.getRaw(), output == null ? null : output.getRaw(), spot_sigma, outline_sigma), device);
    }
    
}
