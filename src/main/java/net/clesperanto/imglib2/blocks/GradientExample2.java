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
import net.imglib2.Interval;
import net.imglib2.algorithm.blocks.*;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.PrimitiveType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;

import java.util.Arrays;
import java.util.function.Function;

import static net.imglib2.util.Util.safeInt;

public class GradientExample2 {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final DeviceJ device = DeviceJ.getDefaultDevice();
        final BlockSupplier<FloatType> blocks = BlockSupplier.of(img).andThen(Tier1.gradientX(device));
        final Img<FloatType> out = BlockAlgoUtils.cellImg(blocks, img.dimensionsAsLongArray(), new int[]{32});

        BdvSource source = BdvFunctions.show(out, "gradientX", Bdv.options().is2D().numRenderingThreads(1));
        source.setDisplayRange(-128, 127);

    }

    public static class Tier1 {

        public static <T extends NativeType<T>>
        Function<BlockSupplier<T>, UnaryBlockOperator<T, FloatType>> gradientX(final DeviceJ device) {
            return s -> {
                final T type = s.getType();
                final int n = s.numDimensions();
                return createGradientXOperator(device, type, n);
            };
        }

        public static <T extends NativeType<T>>
        UnaryBlockOperator<T, FloatType> createGradientXOperator(final DeviceJ device, final T sourceType, final int numDimensions) {
            final GradientX_BlockProcessor_UINT8 proc = new GradientX_BlockProcessor_UINT8(device, numDimensions);
            return new DefaultUnaryBlockOperator<>(sourceType, new FloatType(), numDimensions, numDimensions, proc);
        }

        // TODO: abstract base class for BlockProcessors that upload/download to/from ArrayJ
        private static class GradientX_BlockProcessor_UINT8 extends AbstractBlockProcessor<byte[], float[]> {

            private final DeviceJ device;
            private final TempArrayJ tempArraySrc;
            private final TempArrayJ tempArrayDest;
            private final int[] destSize;

            public GradientX_BlockProcessor_UINT8(final DeviceJ device, final int numDimensions) {
                super(PrimitiveType.BYTE, numDimensions);
                this.device = device;
                this.tempArraySrc = new TempArrayJ(device, DataType.UINT8, MemoryType.BUFFER);
                this.tempArrayDest = new TempArrayJ(device, DataType.FLOAT32, MemoryType.BUFFER);
                this.destSize = new int[numDimensions];
            }

            private GradientX_BlockProcessor_UINT8(final GradientX_BlockProcessor_UINT8 proc) {
                super(proc);
                this.device = proc.device;
                this.tempArraySrc = proc.tempArraySrc.newInstance();
                this.tempArrayDest = proc.tempArrayDest.newInstance();
                this.destSize = new int[proc.destSize.length];
            }

            @Override
            public BlockProcessor<byte[], float[]> independentCopy() {
                return new GradientX_BlockProcessor_UINT8(this);
            }

            @Override
            public void setTargetInterval(Interval interval) {
                Arrays.setAll(destSize, d -> safeInt(interval.dimension(d)));
                Arrays.setAll(sourcePos, d -> interval.min(d) - ((d == 0) ? 1 : 0));
                Arrays.setAll(sourceSize, d -> destSize[d] + ((d == 0) ? 2 : 0));
            }

            @Override
            public void compute(byte[] src, float[] dest) {
                final ArrayJ srcA = tempArraySrc.get(sourceSize);
                final ArrayJ destA = tempArrayDest.get(sourceSize);

                srcA.writeFromArray(src);
                net.clesperanto.kernels.Tier1.gradientX(device, srcA, destA);
                destA.readToArray(dest, new long[]{1, 0, 0}, destSize);
            }
        }
    }
}
