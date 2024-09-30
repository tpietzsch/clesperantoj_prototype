package net.clesperanto.imglib2.blocks;

import bdv.img.n5.BdvN5Format;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvSource;
import ij.IJ;
import ij.ImagePlus;
import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DataType;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.MemoryJ;
import net.clesperanto.imglib2.ImgLib2Converters;
import net.clesperanto.imglib2.ImgLib2DataType;
import net.clesperanto.kernels.Tier1;
import net.imglib2.algorithm.blocks.AbstractBlockSupplier;
import net.imglib2.algorithm.blocks.BlockAlgoUtils;
import net.imglib2.algorithm.blocks.BlockSupplier;
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.fluent.RandomAccessibleIntervalView;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import static net.imglib2.util.Util.int2long;
import static net.imglib2.util.Util.safeInt;
import static net.imglib2.view.fluent.RandomAccessibleIntervalView.Extension.border;

public class GradientExample {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final BlockSupplier<FloatType> blocks = new GradientXSupplier(img);
        final Img<FloatType> out = BlockAlgoUtils.cellImg(blocks, img.dimensionsAsLongArray(), new int[]{32});

        BdvSource source = BdvFunctions.show(out, "gradientX", Bdv.options().is2D());
        source.setDisplayRange(-128, 127);

    }

    static class GradientXSupplier extends AbstractBlockSupplier<FloatType> {

        private final DeviceJ currentDevice;
        private final BlockSupplier<UnsignedByteType> imgBlocks;

        GradientXSupplier(final Img<UnsignedByteType> img) {
            this.currentDevice = DeviceJ.getDefaultDevice();
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
            final ArrayJ srcA = ImgLib2DataType.UINT8.makeAndWriteArrayJ(src, currentDevice, srcSizeL, "buffer");
            final ArrayJ tmpA = ImgLib2DataType.FLOAT32.makeEmptyArrayJ(currentDevice, srcSizeL, "buffer");
            final ArrayJ destA = ImgLib2DataType.FLOAT32.makeEmptyArrayJ(currentDevice, int2long(size), "buffer");
            Tier1.gradientX(currentDevice, srcA, tmpA);
            final int depth = n > 2 ? size[2] : 1;
            Tier1.crop(currentDevice, tmpA, destA, 1, 0, 0, size[0], size[1], depth);

            final int destLen = safeInt(Intervals.numElements(size));
            final int capacity = safeInt((long) destLen * DataType.FLOAT32.getByteSize());
            final ByteBuffer bbuf = ByteBuffer
                    .allocateDirect(capacity)
                    .order(ByteOrder.LITTLE_ENDIAN);
            DataType.FLOAT32.readToBuffer(destA, bbuf);

            final FloatBuffer fbuf = bbuf.asFloatBuffer();
            final float[] fdest = (float[]) dest;
            fbuf.position( 0 );
            fbuf.get( fdest, 0, destLen );

            // TODO
            //  [+] cut appropriate region from img
            //      [+] srcInterval -> (pos, size) +1 on each side in X
            //      [+] allocate byte[] array of appropriate size
            //      [+] imgBlocks.copy(...)
            //  [+] copy to (new, for now) input ArrayJ
            //      [+] copy to ArrayJ (new, for now, but need a TempArray-like thing for ArrayJs soon!)
            //  [+] create gradient output (new, for now): srcInterval size
            //  [+] create crop output (new, for now): targetInterval size
            //  [+] gradient
            //  [+] crop
            //  [+] copy crop output to fdest
            //      [+] read to ByteBuffer
            //      [+] copy ByteBuffer to fdest buffer (using MemCopy)
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
