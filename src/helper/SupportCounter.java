package helper;

import data_structure.ConditionalDatabase;
import data_structure.Predicate;
import data_structure.Support;
import constants.PredicateConstants;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class SupportCounter {
    public static ArrayList<Predicate> getSupportAllPredicate(ArrayList<ArrayList<Predicate>> database,
                                                              ArrayList<Support> dataClasses) {
        HashMap<Integer, Support> counter = new HashMap<>();
        for (int i = 0; i < database.size(); i++) {
            Support dataClass = dataClasses.get(i);
            for (Predicate p: database.get(i)) {
               addToSupport(p, dataClass, counter);
            }
        }

        ArrayList<Predicate> arrPredicate = new ArrayList<>();
        for (Map.Entry<Integer, Support> entry : counter.entrySet()) {
            Support support = entry.getValue();
            arrPredicate.add(new Predicate(entry.getKey(), support.plusSupport, support.negativeSupport));
        }

        return arrPredicate;
    }

    private static void addToSupport(Predicate p, Support dataClass, HashMap<Integer, Support> counter) {
        if (!counter.containsKey(p.id)) {
            counter.put(p.id, new Support(0, 0));
        }
        Support support = counter.get(p.id);
        support.plusSupport += dataClass.plusSupport;
        support.negativeSupport += dataClass.negativeSupport;
        counter.put(p.id, support);
    }

    public static Support getSuppotOfTransaction(ArrayList<Integer> transactions, ConditionalDatabase fullDatabase) {
        Support totalSupport = new Support(0, 0);
        for (Integer transaction: transactions) {
            Support transactionSupport = fullDatabase.dataClasses.get(transaction);
            totalSupport.plusSupport += transactionSupport.plusSupport;
            totalSupport.negativeSupport += transactionSupport.negativeSupport;
        }
        return totalSupport;
    }

    public static Support getSupportOfPattern(ArrayList<Integer> pattern, ArrayList<ArrayList<Predicate>> database, ArrayList<Support> dataClasses) {
        Support support = new Support(0,0);

        for (int i = 0; i < database.size(); i++) {
            ArrayList<Predicate> transaction = database.get(i);
            boolean isContained = true;

            HashSet<Integer> transactionSet = new HashSet<>();
            for (Predicate p: transaction) {
                transactionSet.add(p.id);
            }

            for (Integer predicateId: pattern) {
                if (!transactionSet.contains(predicateId)) {
                    isContained = false;
                    break;
                }
            }

            if (isContained) {
                Support transactionSupport = dataClasses.get(i);
                support.plusSupport += transactionSupport.plusSupport;
                support.negativeSupport += transactionSupport.negativeSupport;
            }
        }
        return support;
    }


}
