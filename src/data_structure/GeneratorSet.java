package data_structure;

import helper.MathHelper;

import java.util.ArrayList;
import java.util.HashMap;

public class GeneratorSet {
    public HashMap<Support, ArrayList<Generator>> GS;

    public GeneratorSet() {
        GS = new HashMap<>();
    }

    public Support getMinimumDSSupport(Support fullDatabaseSupport) {
        Double minDS = -1.0;
        Support minimumSupport = null;

        int fullPositiveSupport = fullDatabaseSupport.plusSupport;
        int fullNegativeSupport = fullDatabaseSupport.negativeSupport;

        for (Support support: GS.keySet()) {
            Double currentDS = MathHelper.discriminativeSignificance(support.plusSupport, support.negativeSupport, fullPositiveSupport, fullNegativeSupport);
            if (minimumSupport == null || currentDS < minDS) {
                minDS = currentDS;
                minimumSupport = support;
            }
        }

        return minimumSupport;
    }

    public Double getMinimumDS(Support fullDatabaseSupport) {
        Double minDS = -1.0;

        int fullPositiveSupport = fullDatabaseSupport.plusSupport;
        int fullNegativeSupport = fullDatabaseSupport.negativeSupport;

        for (Support support: GS.keySet()) {
            Double currentDS = MathHelper.discriminativeSignificance(support.plusSupport, support.negativeSupport, fullPositiveSupport, fullNegativeSupport);
            if (minDS == -1.0 || currentDS < minDS) {
                minDS = currentDS;
            }
        }

        return minDS;
    }

    public void print(Support fullDatabaseSupport) {
        int plusSupport = fullDatabaseSupport.plusSupport;
        int negativeSupport = fullDatabaseSupport.negativeSupport;

        for (Support support: GS.keySet()) {
            System.out.printf("%s : %.4f %s\n", support.toString(), MathHelper.discriminativeSignificance(support.plusSupport, support.negativeSupport,
                    plusSupport, negativeSupport), GS.get(support).toString());
        }
    }

    public void print() {
        for (Support support: GS.keySet()) {
            System.out.printf("%s : %s\n", support.toString(), GS.get(support).toString());
        }
    }
}
