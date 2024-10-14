package net.clesperanto.imglib2.blocks;

import net.clesperanto.core.ArrayJ;
import net.clesperanto.core.DataType;
import net.clesperanto.core.DeviceJ;
import net.clesperanto.core.MemoryType;

import java.lang.ref.WeakReference;

public final class TempArrayJ {

    private final DeviceJ device;
    private final DataType dataType;
    private final MemoryType memoryType;

    private WeakReference<ArrayJ> arrayRef = new WeakReference<>(null);

    public TempArrayJ(final DeviceJ device, final DataType dataType, final MemoryType memoryType) {
        this.device = device;
        this.dataType = dataType;
        this.memoryType = memoryType;
    }

    public TempArrayJ newInstance() {
        return new TempArrayJ(device, dataType, memoryType);
    }

    public ArrayJ get(final int[] size) {
        ArrayJ array = arrayRef.get();
        if (array == null || !sizesMatch(array, size)) {
            if (array != null) {
                array.close();
            }
            array = device.createArray(dataType, memoryType, size);
            arrayRef = new WeakReference<>(array);
        }
        return array;
    }

    private static boolean sizesMatch(final ArrayJ array, final int[] size) {
        final int n = size.length;
        final int w = n > 0 ? size[0] : 1;
        final int h = n > 1 ? size[1] : 1;
        final int d = n > 2 ? size[2] : 1;
        return array.numDimensions() == n && array.width() == w && array.height() == h && array.depth() == d;
    }
}
