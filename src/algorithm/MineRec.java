package algorithm;

import data_structure.*;
import helper.MathHelper;
import helper.MemoryWatcher;

import java.util.ArrayList;

public class MineRec {
    public static void mineRec(ConditionalDatabase fullDatabase, GrTree grTree, int k, int negSup, int sizeLimit, GeneratorSet GS) {
        Support fullDatabaseSupport = fullDatabase.countTotalSupport();
        if (grTree.headTable.size() == 0) {
            return;
        }

        MemoryWatcher.getInstance().ping();

        for (Predicate p: grTree.headTable) {
            ArrayList<Integer> newPrefix = new ArrayList<>(grTree.prefix);
            newPrefix.add(p.id);

            Support patternSupport = fullDatabase.countSupportOfAPattern(newPrefix);

            updateResult(GS, k, patternSupport, fullDatabaseSupport, newPrefix);
        }

        if (grTree.prefix.size() + 1 == sizeLimit) {
            return;
        }

        if (grTree.headTable.size() <= 1) {
            return;
        }

        for (Predicate p: grTree.headTable) {

            ConditionalDatabase newDatabase = new ConditionalDatabase(grTree, p.id);

            newDatabase.removeItemByNegativeSupport(negSup);

            GrTree newGrTree = new GrTree(newDatabase);

            Support prefixSupport = fullDatabase.countSupportOfAPattern(new ArrayList<>(newGrTree.prefix));
            Support unavoidableSupport = newGrTree.countUnavoidableSupport();

            Double upperBound = MathHelper.discriminativeSignificance(
                unavoidableSupport.plusSupport,
                prefixSupport.negativeSupport,
                fullDatabaseSupport.plusSupport,
                fullDatabaseSupport.negativeSupport
            );

            if (GS.GS.keySet().size() == k && upperBound < GS.getMinimumDS(fullDatabaseSupport)) {
                continue;
            }

            MemoryWatcher.getInstance().ping();

            mineRec(fullDatabase, newGrTree, k, negSup, sizeLimit, GS);

            MemoryWatcher.getInstance().ping();
        }
    }

    private static void updateResult(GeneratorSet GS, int k, Support patternSupport, Support fullDatabaseSupport, ArrayList<Integer> newPrefix) {

        MemoryWatcher.getInstance().ping();

        Generator generator = new Generator(newPrefix);
        if (GS.GS.containsKey(patternSupport)) {
            GS.GS.get(patternSupport).add(generator);
        }
        else  {
            GS.GS.put(patternSupport, new ArrayList<>());
            GS.GS.get(patternSupport).add(generator);
        }

        if (GS.GS.keySet().size() > k && k > 0) {
            removeMinDSFromGeneratorSet(GS, fullDatabaseSupport);
        }

        MemoryWatcher.getInstance().ping();
    }

    private static void removeMinDSFromGeneratorSet(GeneratorSet GS, Support fullDatabaseSupport) {
        Support minimumSupport = GS.getMinimumDSSupport(fullDatabaseSupport);

        if (minimumSupport != null) {
            GS.GS.remove(minimumSupport);
        }
    }

}
