package data_structure;

import helper.Converter;
import helper.MemoryWatcher;
import helper.Sorter;
import helper.SupportCounter;

import java.sql.SQLOutput;
import java.util.*;

public class ConditionalDatabase {
    public HashSet<Integer> prefix;
    public ArrayList<ArrayList<Predicate>> database;
    public ArrayList<Support> dataClasses;


    public ConditionalDatabase(ArrayList<ArrayList<Predicate>> database, ArrayList<Support> dataClasses) {
        ArrayList<Predicate> arrPredicate = SupportCounter.getSupportAllPredicate(database, dataClasses);

        // fill database with support
        HashMap<Integer, Support> counter = Converter.convertPredicateArrayToHashMap(arrPredicate);
        Converter.fillDatabaseWithSupport(database, counter);

        // copy database
        this.database = new ArrayList<>();
        for (ArrayList<Predicate> arr: database) {
            this.database.add(new ArrayList<>(arr));
        }

        // copy dataClasses
        this.dataClasses = new ArrayList<>();
        this.dataClasses.addAll(dataClasses);

        // empty prefix
        prefix = new HashSet<>();

        MemoryWatcher.getInstance().ping();
    }

    public ConditionalDatabase(GrTree grTree, Integer prefixAddition) {
        this.prefix = new HashSet<>();
        this.prefix.addAll(grTree.prefix);
        this.prefix.add(prefixAddition);

        database = new ArrayList<>();
        dataClasses = new ArrayList<>();

        createDatabase(grTree, prefixAddition);

        MemoryWatcher.getInstance().ping();
    }

    public Support countSupportOfAPattern(ArrayList<Integer> pattern) {
        return SupportCounter.getSupportOfPattern(pattern, database, dataClasses);
    }

    public Support countTotalSupport() {
        Support totalSupport = new Support(0, 0);
        for (Support support: dataClasses) {
            totalSupport.plusSupport += support.plusSupport;
            totalSupport.negativeSupport += support.negativeSupport;
        }

        return totalSupport;
    }

    private void createDatabase(GrTree grTree, Integer prefixAddition) {
        for (Map.Entry<Integer, GrTree.GrNode> entry: grTree.root.children.entrySet()) {
            GrTree.GrNode child = entry.getValue();
            traverse(child, prefixAddition, new ArrayList<>());
        }
    }

    public void removeItemByNegativeSupport(int negSup) {
        ArrayList<ArrayList<Predicate>> newDatabase = new ArrayList<>();
        ArrayList<Support> newDataClasses = new ArrayList<>();

        for (int i = 0; i < dataClasses.size(); i++) {
            if (dataClasses.get(i).negativeSupport >= negSup) {
                newDatabase.add(database.get(i));
                newDataClasses.add(dataClasses.get(i));
            }
        }

        database = newDatabase;
        dataClasses = newDataClasses;
    }

    private void traverse(GrTree.GrNode grNode, Integer prefixAddition, ArrayList<Predicate> path) {
        if (grNode.predicate.id == prefixAddition && path.size() > 0) {
            database.add(new ArrayList<>(path));

            Predicate predicate = grNode.predicate;
            Support support = new Support(predicate.plusSupport, predicate.negativeSupport);

            dataClasses.add(support);
            return;
        }

        for (Map.Entry<Integer, GrTree.GrNode> entry: grNode.children.entrySet()) {
            path.add(grNode.predicate);
            traverse(entry.getValue(), prefixAddition, path);
            path.remove(path.size()-1);
        }

        MemoryWatcher.getInstance().ping();
    }

    public void printDatabase() {
        for (int i = 0; i < database.size(); i++) {
            System.out.printf("%s %s\n", Arrays.toString(database.get(i).toArray()), dataClasses.get(i).toString());
        }
    }
}
