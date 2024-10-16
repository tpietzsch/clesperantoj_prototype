package net.clesperanto.imglib2.blocks;

import bdv.cache.SharedQueue;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvSource;
import bdv.util.volatiles.VolatileViews;
import bdv.viewer.DisplayMode;
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
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.*;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import java.util.Arrays;
import java.util.function.Function;

public class GaussianBlurExample2 {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/e002_stack_fused-8bit.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final BdvSource bdv = BdvFunctions.show(
                img,
                "img",
                Bdv.options() );
        bdv.setColor( new ARGBType( 0xffffff ) );
        bdv.setDisplayRange( 0, 255 );
        bdv.getBdvHandle().getViewerPanel().setDisplayMode( DisplayMode.SINGLE );

        final DeviceJ device = DeviceJ.getDefaultDevice();
        final BlockSupplier<FloatType> blocks = BlockSupplier
                .of(Views.extendMirrorDouble(img))
                .andThen(Tier1.gaussianBlur(device, 5, 5, 5));

        final long[] dimensions = img.dimensionsAsLongArray();
        final int[] cellDimensions = { 64, 64, 64 };
        final Img<FloatType> convolved = BlockAlgoUtils.cellImg(
                blocks,
                dimensions,
                cellDimensions );

        final BdvSource out = BdvFunctions.show(
                VolatileViews.wrapAsVolatile( convolved, new SharedQueue( 8, 1 ) ),
                "convolve",
                Bdv.options()
                        .addTo( bdv )
        );
        out.setDisplayRange( 0, 255 );
        out.setColor( new ARGBType( 0x00ff00 ) );
    }

    public static void main3(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/e002_stack_fused-8bit.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final DeviceJ device = DeviceJ.getDefaultDevice();
//        final BlockSupplier<FloatType> blocks = BlockSupplier.of(img).andThen(Tier1.gaussianBlur(device, 5, 5, 5));

        final BlockSupplier<FloatType> blocks = BlockSupplier.of(img)
                .andThen(Tier1.gaussianBlur(device, 5, 5, 5))
                .andThen(Tier1.gradientX(device));

        final long[] dimensions = img.dimensionsAsLongArray();
        final int[] cellDimensions = { 64, 64, 64 };
        final Img<FloatType> out = BlockAlgoUtils.cellImg(
                blocks,
                dimensions,
                cellDimensions );

        BdvSource source = BdvFunctions.show(VolatileViews.wrapAsVolatile(out, new SharedQueue(8, 1)), "gaussianBlur");
        source.setDisplayRange(-128, 127);

    }

    public static void main2(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final DeviceJ device = DeviceJ.getDefaultDevice();
        final BlockSupplier<FloatType> blocks = BlockSupplier.of(img).andThen(Tier1.gaussianBlur(device, 5, 5));
        final Img<FloatType> out = BlockAlgoUtils.cellImg(blocks, img.dimensionsAsLongArray(), new int[]{32});

        BdvSource source = BdvFunctions.show(VolatileViews.wrapAsVolatile(out), "gaussianBlur", Bdv.options().is2D().numRenderingThreads(1));
        source.setDisplayRange(0, 255);

    }

    public static class Tier1 {

        public static <T extends NativeType<T>>
        Function<BlockSupplier<T>, UnaryBlockOperator<T, FloatType>> gaussianBlur(final DeviceJ device, final float... sigmas) {
            return s -> {
                final T type = s.getType();
                final int n = s.numDimensions();
                return createGaussianBlurOperator(device, type, n, sigmas);
            };
        }

        public static <T extends NativeType<T>>
        UnaryBlockOperator<T, FloatType> createGaussianBlurOperator(final DeviceJ device, final T sourceType, final int numDimensions, final float... sigmas) {
            final GaussianBlur_BlockProcessor_UINT8 proc = new GaussianBlur_BlockProcessor_UINT8(device, numDimensions, sigmas);
            return new DefaultUnaryBlockOperator<>(sourceType, new FloatType(), numDimensions, numDimensions, proc);
        }

        // TODO: abstract base class for BlockProcessors that upload/download to/from ArrayJ
        private static class GaussianBlur_BlockProcessor_UINT8 extends AbstractClicBlockProcessor<byte[], float[]> {

            private final float[] sigmas;
            private final int[] padWidth;
            private final int[] padOffset;

            public GaussianBlur_BlockProcessor_UINT8(final DeviceJ device, final int numDimensions, final float... sigmas) {
                super(device, DataType.UINT8, DataType.FLOAT32, numDimensions, numDimensions);
                this.sigmas = new float[3];
                this.sigmas[0] = sigmas.length > 0 ? sigmas[0] : 0;
                this.sigmas[1] = sigmas.length > 1 ? sigmas[1] : 0;
                this.sigmas[2] = sigmas.length > 2 ? sigmas[2] : 0;
                padWidth = new int[numDimensions];
                padOffset = new int[numDimensions];
                for (int d = 0; d < numDimensions; d++) {
                    if (sigmas[d] == 0) {
                        padWidth[d] = 0;
                        padOffset[d] = 0;
                    } else {
                        padWidth[d] = sigma2kernelsize(sigmas[d]) - 1;
                        padOffset[d] = padWidth[d] / 2;
                    }
                }
            }

            // TODO: The following is copied from CLIc C++ code. Ultimately, we
            //       should move the target-to-source-interval mapping to CLIc.
            private static int sigma2kernelsize(final float sigma) {
                final int rad = (int) (sigma * 8.0);
                return (rad % 2 == 0) ? rad + 1 : rad;
            }

            private GaussianBlur_BlockProcessor_UINT8(final GaussianBlur_BlockProcessor_UINT8 proc) {
                super(proc);
                this.sigmas = proc.sigmas;
                this.padWidth = proc.padWidth;
                this.padOffset = proc.padOffset;
            }

            @Override
            public BlockProcessor<byte[], float[]> independentCopy() {
                return new GaussianBlur_BlockProcessor_UINT8(this);
            }

            @Override
            protected void setTargetInterval(long[] pos, int[] size, long[] sourcePos, int[] sourceSize, long[] targetOffset, int[] targetSize) {
                Arrays.setAll(sourcePos, d -> pos[d] - padOffset[d]);
                Arrays.setAll(sourceSize, d -> size[d] + padWidth[d]);
                Arrays.setAll(targetOffset, d -> padOffset[d]);
                Arrays.setAll(targetSize, d -> size[d]);
            }

            @Override
            protected void compute(ArrayJ src, ArrayJ dest) {
                net.clesperanto.kernels.Tier1.gaussianBlur(device, src, dest, sigmas[0], sigmas[1], sigmas[2]);
            }
        }

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

            final GradientX_BlockProcessor proc = new GradientX_BlockProcessor(device, dataType(sourceType), numDimensions);
            return new DefaultUnaryBlockOperator<>(sourceType, new FloatType(), numDimensions, numDimensions, proc);
        }

        // TODO: abstract base class for BlockProcessors that upload/download to/from ArrayJ
        private static class GradientX_BlockProcessor<I> extends AbstractClicBlockProcessor<I, float[]> {

            private final int[] padWidth;
            private final int[] padOffset;

            public GradientX_BlockProcessor(final DeviceJ device, final DataType sourceDataType, final int numDimensions) {
                super(device, sourceDataType, DataType.FLOAT32, numDimensions, numDimensions);
                padWidth = new int[numDimensions];
                padOffset = new int[numDimensions];
                final int gradientDimension = 0;
                Arrays.setAll(padOffset, d -> ((d == gradientDimension) ? 1 : 0));
                Arrays.setAll(padWidth, d -> ((d == gradientDimension) ? 2 : 0));

            }

            private GradientX_BlockProcessor(final GradientX_BlockProcessor<I> proc) {
                super(proc);
                this.padWidth = proc.padWidth;
                this.padOffset = proc.padOffset;
            }

            @Override
            public BlockProcessor<I, float[]> independentCopy() {
                return new GradientX_BlockProcessor<>(this);
            }

            @Override
            protected void setTargetInterval(long[] pos, int[] size, long[] sourcePos, int[] sourceSize, long[] targetOffset, int[] targetSize) {
                Arrays.setAll(sourcePos, d -> pos[d] - padOffset[d]);
                Arrays.setAll(sourceSize, d -> size[d] + padWidth[d]);
                Arrays.setAll(targetOffset, d -> padOffset[d]);
                Arrays.setAll(targetSize, d -> size[d]);
            }

            @Override
            protected void compute(ArrayJ src, ArrayJ dest) {
                net.clesperanto.kernels.Tier1.gradientX(device, src, dest);
            }
        }
    }







    public static abstract class AbstractClicBlockProcessor< I, O > extends AbstractBlockProcessor< I, O >
    {
        private final TempArrayJ tempArraySrc;
        private final TempArrayJ tempArrayDest;

        private final long[] targetOffset;
        private final int[] targetSize;

        protected final DeviceJ device;

        protected AbstractClicBlockProcessor(
                final DeviceJ device,
                final DataType sourceDataType,
                final DataType targetDataType,
                final int numSourceDimensions,
                final int numTargetDimensions) {
            super(primitiveTypeFor(sourceDataType), checkNumDimensions(numSourceDimensions));
            checkNumDimensions(numTargetDimensions);
            this.device = device;
            this.tempArraySrc = new TempArrayJ(device, sourceDataType, MemoryType.BUFFER);
            this.tempArrayDest = new TempArrayJ(device, targetDataType, MemoryType.BUFFER);
            this.targetOffset = new long[numTargetDimensions];
            this.targetSize = new int[numTargetDimensions];
        }

        protected AbstractClicBlockProcessor(final AbstractClicBlockProcessor<I, O> proc) {
            super(proc);
            this.device = proc.device;
            this.tempArraySrc = proc.tempArraySrc.newInstance();
            this.tempArrayDest = proc.tempArrayDest.newInstance();
            final int numTargetDimensions = proc.targetOffset.length;
            this.targetOffset = new long[numTargetDimensions];
            this.targetSize = new int[numTargetDimensions];
        }

        @Override
        public final void setTargetInterval(Interval interval) {
            // we extract the interval into pos/size arrays, because eventually
        }

        @Override
        public final void setTargetInterval( long[] pos, int[] size )
        {
            setTargetInterval(pos, size, sourcePos, sourceSize, targetOffset, targetSize);
        }

        @Override
        public final void compute(I src, O dest) {
            final ArrayJ srcA = tempArraySrc.get(sourceSize);
            final ArrayJ destA = tempArrayDest.get(sourceSize);
            srcA.writeFromArray(src);
            compute(srcA, destA);
            destA.readToArray(dest, targetOffset, targetSize);
        }

        /**
         *
         * @param pos input
         * @param size input
         * @param sourcePos output, to be filled
         * @param sourceSize output, to be filled
         * @param targetOffset output, to be filled
         * @param targetSize output, to be filled
         */
        protected abstract void setTargetInterval( long[] pos, int[] size, long[] sourcePos, int[] sourceSize, long[] targetOffset, int[] targetSize );

        /**
         * TODO javadoc
         */
        protected abstract void compute( ArrayJ src, ArrayJ dest );

        private static int checkNumDimensions(final int numDimensions) {
            if ( numDimensions > 3 ) {
                throw new IllegalArgumentException("clEsperanto supports at most 3 dimensions");
            }
            return numDimensions;
        }
    }

    /**
     * TODO: Probably should be moved to some utils class
     * TODO: javadoc
     */
    public static PrimitiveType primitiveTypeFor(DataType dataType) {
        switch (dataType) {
            case INT8:
            case UINT8:
                return PrimitiveType.BYTE;
            case INT16:
            case UINT16:
                return PrimitiveType.SHORT;
            case INT32:
            case UINT32:
                return PrimitiveType.INT;
            case FLOAT32:
            default:
                return PrimitiveType.FLOAT;
        }
    }

    /**
     * TODO: This is a duplicate of {@link net.clesperanto.imglib2.ImgLib2Converters#dataType(NativeType)}
     * TODO: Probably should be moved to some utils class
     * TODO: javadoc
     */
    public static <T extends NativeType<T>> DataType dataType(final T type) {
        if (type instanceof FloatType)
            return DataType.FLOAT32;
        else if (type instanceof IntType)
            return DataType.INT32;
        else if (type instanceof UnsignedIntType)
            return DataType.UINT32;
        else if (type instanceof ShortType)
            return DataType.INT16;
        else if (type instanceof UnsignedShortType)
            return DataType.UINT16;
        else if (type instanceof ByteType)
            return DataType.INT8;
        else if (type instanceof UnsignedByteType)
            return DataType.UINT8;
        else
            throw new IllegalArgumentException("Type not supported: " + type.getClass().getName());
    }

}
