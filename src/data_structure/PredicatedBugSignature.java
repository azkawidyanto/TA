package data_structure;

import helper.MathHelper;
import helper.SupportCounter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

public class PredicatedBugSignature {
    public HashMap<ArrayList<Integer>, ArrayList<Generator>> bugSignature;
    public ArrayList<BugSignatureDSPair> bugSignatureDSPairs;

    public PredicatedBugSignature() {
        bugSignature = new HashMap<>();
    }

    public void addBugSignatureToDSPairs(ConditionalDatabase fullDatabase) {
        bugSignatureDSPairs = new ArrayList<>();
        Support fullDatabaseSupport = fullDatabase.countTotalSupport();

        for (ArrayList<Integer> transaction: bugSignature.keySet()) {
            Support transactionSupport = SupportCounter.getSuppotOfTransaction(transaction, fullDatabase);
            Double ds = MathHelper.discriminativeSignificance(
                transactionSupport.plusSupport,
                transactionSupport.negativeSupport,
                fullDatabaseSupport.plusSupport,
                fullDatabaseSupport.negativeSupport
            );

            bugSignatureDSPairs.add(new BugSignatureDSPair(ds, bugSignature.get(transaction)));
        }

        bugSignatureDSPairs.sort(new Comparator<BugSignatureDSPair>() {
            @Override
            public int compare(BugSignatureDSPair pair1, BugSignatureDSPair pair2) {
                if (pair1.ds.equals(pair2.ds)) {
                    return 0;
                }
                else if (pair1.ds < pair2.ds) {
                    return 1;
                }
                else {
                    return -1;
                }
            }
        });
    }

    public void print() {
        for (BugSignatureDSPair pair: bugSignatureDSPairs) {
            System.out.printf("%s : %.4f\n", pair.generators.toString(), pair.ds);
        }
    }

    public void clean(Double limit) {
        ArrayList<BugSignatureDSPair> newBugSignatureDSPairs = new ArrayList<>();

        for (BugSignatureDSPair pair: bugSignatureDSPairs) {
            if (pair.ds >= limit) {
                newBugSignatureDSPairs.add(pair);
            }
        }

        bugSignatureDSPairs = newBugSignatureDSPairs;
    }
}
