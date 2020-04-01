package helper;

import java.lang.Math;

public class MathHelper {
    private static double log2(double x) {
        if (x == 0) {
            return -1*Double.MAX_VALUE;
        }
        return Math.log(x) / Math.log(2);
    }

    private static double h(double a, double b) {
        double part_a = (a/(a+b)) * log2(a/(a+b));
        double part_b = (b/(a+b)) * log2(b/ (a+b));
        return - part_a - part_b;
    }

    private static double informationGain(double p, double n, double plus_size, double negative_size) {
        double total = plus_size + negative_size;

        double total_h = h(plus_size, negative_size);
        double current_h = ((p+n) / total) * h(p, n);
        double total_without_current_h = ((total - p - n) / total) * h(plus_size-p, negative_size-n);


        return total_h - current_h -total_without_current_h;
    }

    public static double discriminativeSignificance(double p, double n, double plus_size, double negative_size) {
        boolean is_dominant_negative = Math.abs((n/negative_size)) > Math.abs((p/plus_size));
        if (is_dominant_negative) {
            return informationGain(p, n, plus_size, negative_size);
        } else {
            return 0;
        }
    }
}
