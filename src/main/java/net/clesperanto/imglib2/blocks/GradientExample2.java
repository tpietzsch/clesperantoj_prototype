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
import net.imglib2.algorithm.blocks.*;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.PrimitiveType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;

import java.util.Arrays;

import static net.imglib2.util.Util.safeInt;
import static net.imglib2.view.fluent.RandomAccessibleIntervalView.Extension.border;

public class GradientExample2 {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final BlockSupplier<FloatType> blocks = new GradientXSupplier(img);
        final Img<FloatType> out = BlockAlgoUtils.cellImg(blocks, img.dimensionsAsLongArray(), new int[]{32});

        BdvSource source = BdvFunctions.show(out, "gradientX", Bdv.options().is2D());
        source.setDisplayRange(-128, 127);

    }

    static class GradientX_BlockProcessor_UINT8 extends AbstractBlockProcessor<byte[], float[]> {

        private final DeviceJ device;

        public GradientX_BlockProcessor_UINT8(final DeviceJ device, final int numDimensions) {
            super(PrimitiveType.BYTE, numDimensions);
            this.device = device;
        }

        private GradientX_BlockProcessor_UINT8(final GradientX_BlockProcessor_UINT8 proc) {
            super(proc);
            this.device = proc.device;
        }

        @Override
        public BlockProcessor<byte[], float[]> independentCopy() {
            return new GradientX_BlockProcessor_UINT8(this);
        }

        @Override
        public void compute(byte[] src, float[] dest) {
            // TODO
            //   next: TempArrayJ --> holds an ArrayJ
            //   [ ] how is cle::Array released?
        }
    }

    static class GradientXSupplier extends AbstractBlockSupplier<FloatType> {

        private final DeviceJ currentDevice;
        private final BlockSupplier<UnsignedByteType> imgBlocks;

        GradientXSupplier(final Img<UnsignedByteType> img) {
            this.currentDevice = DeviceJ.getDefaultDevice();
            System.out.println("currentDevice.getInfo() = " + currentDevice.getInfo());
            System.out.println("currentDevice.getBackend() = " + currentDevice.getBackend());
            this.imgBlocks = BlockSupplier.of(img.view().extend(border()));
        }

        private GradientXSupplier(final GradientXSupplier s) {
            this.currentDevice = s.currentDevice;
            this.imgBlocks = s.imgBlocks.independentCopy();
        }

        @Override
        public void copy(long[] pos, Object dest, int[] size) {

            final int n = numDimensions();
            final long[] srcPos = new long[n];
            Arrays.setAll(srcPos, d -> pos[d] - ((d == 0) ? 1 : 0));
            final int[] srcSize = new int[n];
            Arrays.setAll(srcSize, d -> size[d] + ((d == 0) ? 2 : 0));
            final long[] srcSizeL = Util.int2long(srcSize);

            final byte[] src = new byte[safeInt(Intervals.numElements(srcSize))]; // TODO: should use TempArray
            imgBlocks.copy(srcPos, src, srcSize);

            // TODO: make a TempArray-like thing for ArrayJ
            final ArrayJ srcA = new ArrayJ(srcSize, currentDevice, DataType.UINT8, MemoryType.BUFFER);
//            final ArrayJ srcA = currentDevice.newArray( DataType.UINT8, jclic.MTypeJ.BUFFER, srcSize );
//            final ArrayJ srcA = currentDevice.createArray( DataType.UINT8, jclic.MTypeJ.BUFFER, srcSize );
            srcA.writeFromArray(src);
            final ArrayJ destA = new ArrayJ(srcSize, currentDevice, DataType.FLOAT32, MemoryType.BUFFER);
            Tier1.gradientX(currentDevice, srcA, destA);

            destA.readToArray(dest, new long[]{1, 0, 0}, size);
        }

        @Override
        public BlockSupplier<FloatType> independentCopy() {
            return new GradientXSupplier(this);
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
