package data_structure;

import java.util.ArrayList;

public class BugSignaturePair {

    public ArrayList<Integer> transaction;
    public ArrayList<Generator> generators;

    public BugSignaturePair(ArrayList<Integer> transaction, ArrayList<Generator> generators) {
        this.transaction = new ArrayList<>(transaction);
        this.generators = new ArrayList<>(generators);
    }

    @Override
    public String toString() {
        return String.format("%s : %s", transaction.toString(), generators.toString());
    }
}
