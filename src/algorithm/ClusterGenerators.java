package algorithm;

import data_structure.*;
import helper.MathHelper;
import helper.MemoryWatcher;
import helper.SupportCounter;

import java.util.*;

public class ClusterGenerators {
    public static PredicatedBugSignature ClusterGeneratorToPredicatedBugSignature(ConditionalDatabase fullDatabase, GeneratorSet generatorSet, int k) {
        PredicatedBugSignature predicatedBugSignature = new PredicatedBugSignature();

        for (Map.Entry<Support, ArrayList<Generator>> entry: generatorSet.GS.entrySet()) {
            for (Generator generator: entry.getValue()) {
                ArrayList<Integer> tx = TX.countTX(fullDatabase, generator.generator);

                if (!predicatedBugSignature.bugSignature.containsKey(tx)) {
                    predicatedBugSignature.bugSignature.put(tx, new ArrayList<>());
                }
                predicatedBugSignature.bugSignature.get(tx).add(generator);
            }
        }

        if (predicatedBugSignature.bugSignature.size() > k && k > 0) {
            ArrayList<BugSignaturePair> sortedBugSignatures = sortBugSignatureBasedOnDS(predicatedBugSignature, fullDatabase);
            predicatedBugSignature.bugSignature = new HashMap<>();

            for (int i = 0; i < k; i+= 1) {
                predicatedBugSignature.bugSignature.put(sortedBugSignatures.get(i).transaction, sortedBugSignatures.get(i).generators);
            }
        }

        MemoryWatcher.getInstance().ping();

        return predicatedBugSignature;
    }



    private static ArrayList<BugSignaturePair> sortBugSignatureBasedOnDS(PredicatedBugSignature predicatedBugSignature, ConditionalDatabase fullDatabase) {
        Support fullDatabaseSupport = fullDatabase.countTotalSupport();

        ArrayList<BugSignaturePair> bugSignaturePairs = new ArrayList<>();
        for (ArrayList<Integer> transaction: predicatedBugSignature.bugSignature.keySet()) {
            bugSignaturePairs.add(new BugSignaturePair(transaction, predicatedBugSignature.bugSignature.get(transaction)));
        }

        bugSignaturePairs.sort(new Comparator<BugSignaturePair>() {
            @Override
            public int compare(BugSignaturePair bp1, BugSignaturePair bp2) {
                Support support1 = SupportCounter.getSuppotOfTransaction(bp1.transaction, fullDatabase);
                Double DS1 =  MathHelper.discriminativeSignificance(
                        support1.plusSupport,
                        support1.negativeSupport,
                        fullDatabaseSupport.plusSupport,
                        fullDatabaseSupport.negativeSupport
                );

                Support support2 = SupportCounter.getSuppotOfTransaction(bp2.transaction, fullDatabase);
                Double DS2 =  MathHelper.discriminativeSignificance(
                        support2.plusSupport,
                        support2.negativeSupport,
                        fullDatabaseSupport.plusSupport,
                        fullDatabaseSupport.negativeSupport
                );

                if (DS1.equals(DS2)) {
                    return 0;
                } else if (DS1 > DS2) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });

        MemoryWatcher.getInstance().ping();

        return bugSignaturePairs;
    }

}
