package net.clesperanto;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.MemoryJ;
import net.clesperanto.imglib2.ImgLib2Converters;
import net.clesperanto.kernels.Tier1;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class BlkExample {

    public static void main(String[] args) {

        final String fn = "/Users/pietzsch/workspace/data/DrosophilaWing.tif";
        final ImagePlus imp = IJ.openImage(fn);
        final Img<UnsignedByteType> img = ImageJFunctions.wrapByte(imp);

        final DeviceJ currentDevice = DeviceJ.getDefaultDevice();

        final ArrayJ input = ImgLib2Converters.copyImgLib2ToArrayJ(img, currentDevice, "buffer");
        System.out.println("input = " + input);

        ArrayJ output2 = MemoryJ.makeFloatBuffer(currentDevice,
                img.dimension(0),
                img.dimension(1)-100,
                1,
                2,
                "buffer");
        System.out.println("output2 = " + output2);

//        final ArrayJ output = Tier1.transposeXy(currentDevice, input, null);
//        final ArrayJ output = Tier1.gradientX(currentDevice, input, null);
        final ArrayJ output = Tier1.gradientX(currentDevice, input, output2);
        System.out.println("output = " + output);

        final Img<UnsignedByteType> img2 = ImgLib2Converters.copyArrayJToImgLib2(output);

        new ImageJ();
        ImageJFunctions.show(img);
        ImageJFunctions.show(img2);
    }
}
