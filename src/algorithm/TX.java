package algorithm;

import data_structure.ConditionalDatabase;
import data_structure.Predicate;

import java.util.ArrayList;
import java.util.HashSet;

public class TX {
    public static ArrayList<ArrayList<Integer>> convertDatabaseToInteger(ArrayList<ArrayList<Predicate>> database) {
        ArrayList<ArrayList<Integer>> databaseInteger = new ArrayList<>();

        for (ArrayList<Predicate> arrPredicate: database) {
            ArrayList<Integer> currentList = new ArrayList<>();
            for (Predicate p: arrPredicate) {
                currentList.add(p.id);
            }
            databaseInteger.add(currentList);
        }

        return databaseInteger;
    }

    public static ArrayList<Integer> convertArrPredicateToInteger(ArrayList<Predicate> arrPredicate) {
        ArrayList<Integer> arrInteger = new ArrayList<>();

        for (Predicate p: arrPredicate) {
            arrInteger.add(p.id);
        }

        return arrInteger;
    }

    public static boolean checkAllElementContainedInHashSet(HashSet<Integer> integerSet, ArrayList<Integer> arrInteger) {
        for (Integer integer: arrInteger) {
            if (!integerSet.contains(integer)) {
                return false;
            }
        }
        return true;
    }

    public static ArrayList<Integer> countTX(ConditionalDatabase conditionalDatabase, ArrayList<Integer> arrInteger) {
        ArrayList<ArrayList<Predicate>> database = conditionalDatabase.database;

        ArrayList<Integer> result = new ArrayList<>();

        ArrayList<ArrayList<Integer>> databaseInteger = convertDatabaseToInteger(database);

        for (int i = 0;i < databaseInteger.size(); i++) {
            ArrayList<Integer> transaction = databaseInteger.get(i);
            HashSet<Integer> integerSet = new HashSet<>(transaction);
            boolean contained = checkAllElementContainedInHashSet(integerSet, arrInteger);

            if (contained) {
                result.add(i);
            }
        }

        return result;
    }

    public static Integer countSuppot(ConditionalDatabase conditionalDatabase, ArrayList<Integer> arrInteger) {
        ArrayList<Integer> tx = countTX(conditionalDatabase, arrInteger);
        return tx.size();
    }
}
