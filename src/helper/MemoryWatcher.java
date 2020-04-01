package helper;

public class MemoryWatcher {
    private long initialFreeMemory;
    private long maxMemoryUsed;
    private static  MemoryWatcher instance;

    private MemoryWatcher() {
        initialFreeMemory = Runtime.getRuntime().freeMemory();
        maxMemoryUsed = 0;
    }

    public void ping() {
        long currentMemoryUsed = initialFreeMemory - Runtime.getRuntime().freeMemory();
        maxMemoryUsed = Math.max(maxMemoryUsed, currentMemoryUsed);
    }

    public long getMaxMemoryUsed() {
        return maxMemoryUsed;
    }
    public static MemoryWatcher getInstance() {
        if (instance == null) {
            instance = new MemoryWatcher();
        }
        return instance;
    }
}
