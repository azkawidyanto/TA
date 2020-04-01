package helper;

import data_structure.Predicate;
import data_structure.Support;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Converter {
    public static void convertInputToData(String[] input, ArrayList<ArrayList<Predicate>> database,
                                          ArrayList<Support> dataClasses) {
        for (String s: input) {
            String[] splitedString = s.split("\\s+");
            if (splitedString.length == 1) {
                Integer stringIntegerValue = Integer.valueOf(splitedString[0]);
                Support support = new Support(stringIntegerValue);
                dataClasses.add(support);
            } else {
                ArrayList<Predicate> arrPredicate = new ArrayList<>();
                for (int i = 0; i < splitedString.length - 1; i++) {
                    arrPredicate.add(new Predicate(Integer.valueOf(splitedString[i])));
                }
                database.add(arrPredicate);
                Integer stringIntegerValue = Integer.valueOf(splitedString[splitedString.length - 1]);
                Support support = new Support(stringIntegerValue);
                dataClasses.add(support);
            }
        }

        MemoryWatcher.getInstance().ping();
    }

    public static HashMap<Integer, Support> convertPredicateArrayToHashMap(ArrayList<Predicate> arrPredicate) {
        HashMap<Integer, Support> counter = new HashMap<>();

        for (Predicate p: arrPredicate) {
            counter.put(p.id, new Support(p.plusSupport, p.negativeSupport));
        }

        MemoryWatcher.getInstance().ping();

        return counter;
    }

    public static void fillDatabaseWithSupport(ArrayList<ArrayList<Predicate>> database,
                                               HashMap<Integer, Support> counter) {
        for (ArrayList<Predicate> arrPredicate: database) {
            for (Predicate p: arrPredicate) {
                Support support = counter.get(p.id);
                p.plusSupport = support.plusSupport;
                p.negativeSupport = support.negativeSupport;
            }
        }

        MemoryWatcher.getInstance().ping();
    }

    public static HashSet<Integer> convertBugPredicateInputToHashSet(String[] input) {
        HashSet<Integer> bugPredicates = new HashSet<>();

        if (input.length < 1) {
            return bugPredicates;
        }

        for (String row: input) {
            bugPredicates.add(Integer.valueOf(row));
        }

        return bugPredicates;
    }
}

