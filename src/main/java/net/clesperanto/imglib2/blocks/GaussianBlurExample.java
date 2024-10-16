package net.clesperanto.imglib2.blocks;

import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvSource;
import ij.IJ;
import ij.ImagePlus;
import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DataType;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.MemoryType;
import net.clesperanto.kernels.Tier1;
import net.imglib2.algorithm.blocks.AbstractBlockSupplier;
import net.imglib2.algorithm.blocks.BlockAlgoUtils;
import net.imglib2.algorithm.blocks.BlockSupplier;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;

import java.util.Arrays;

import static net.imglib2.util.Util.safeInt;
import static net.imglib2.view.fluent.RandomAccessibleIntervalView.Extension.border;

public class GaussianBlurExample {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final BlockSupplier<FloatType> blocks = new GaussianBlurSupplier(img, 5, 5);
        final Img<FloatType> out = BlockAlgoUtils.cellImg(blocks, img.dimensionsAsLongArray(), new int[]{32});

        BdvSource source = BdvFunctions.show(out, "gaussianBlur", Bdv.options().is2D());
        source.setDisplayRange(0, 255);

    }

    // numDimensions in {1,2,3}

    static class GaussianBlurSupplier extends AbstractBlockSupplier<FloatType> {

        private final DeviceJ device;
        private final BlockSupplier<UnsignedByteType> imgBlocks;
        private final float[] sigmas;

        GaussianBlurSupplier(final Img<UnsignedByteType> img, final float... sigmas) {
            device = DeviceJ.getDefaultDevice();
            imgBlocks = BlockSupplier.of(img.view().extend(border()));
            this.sigmas = new float[3];
            this.sigmas[0] = sigmas.length > 0 ? sigmas[0] : 0;
            this.sigmas[1] = sigmas.length > 1 ? sigmas[1] : 0;
            this.sigmas[2] = sigmas.length > 2 ? sigmas[2] : 0;
        }

        private GaussianBlurSupplier(final GaussianBlurSupplier s) {
            this.device = s.device;
            this.sigmas = s.sigmas;
            this.imgBlocks = s.imgBlocks.independentCopy();
        }

        @Override
        public void copy(long[] pos, Object dest, int[] size) {

            final int n = numDimensions();
            final long[] srcPos = new long[n];
            final int[] srcSize = new int[n];
            final long[] destOffset = new long[n];
            for (int d = 0; d < n; d++) {
                if (sigmas[d] == 0) {
                    srcPos[d] = pos[d];
                    srcSize[d] = size[d];
                    destOffset[d] = 0;
                } else {
                    final int kernelWidth = sigma2kernelsize(sigmas[d]);
                    final int kernelCenter = (kernelWidth - 1) / 2;
                    srcPos[d] = pos[d] - kernelCenter;
                    srcSize[d] = size[d] + kernelWidth;
                    destOffset[d] = kernelCenter;
                }
            }

            final byte[] src = new byte[safeInt(Intervals.numElements(srcSize))]; // TODO: should use TempArray
            imgBlocks.copy(srcPos, src, srcSize);

            // dataTypes: src array can be any type. dst array will be FLOAT32
            final ArrayJ srcA = new ArrayJ(srcSize, device, DataType.UINT8, MemoryType.BUFFER);
            srcA.writeFromArray(src);
            final ArrayJ destA = new ArrayJ(srcSize, device, DataType.FLOAT32, MemoryType.BUFFER);
            Tier1.gaussianBlur(device, srcA, destA, sigmas[0], sigmas[1], sigmas[2]);

            destA.readToArray(dest, destOffset, size);
        }

        // TODO: The following is copied from CLIc C++ code. Ultimately, we
        //       should move the target-to-source-interval mapping to CLIc.
        private static int sigma2kernelsize(final float sigma) {
            final int rad = (int) (sigma * 8.0);
            return (rad % 2 == 0) ? rad + 1 : rad;
        }

        @Override
        public BlockSupplier<FloatType> independentCopy() {
            return new GaussianBlurSupplier(this);
        }

        @Override
        public int numDimensions() {
            return imgBlocks.numDimensions();
        }

        private static final FloatType type = new FloatType();

        @Override
        public FloatType getType() {
            return type;
        }
    }

}
