package data_structure;

import java.util.ArrayList;

public class BugSignatureDSPair {
    public Double ds;
    public ArrayList<Generator> generators;

    public BugSignatureDSPair(Double ds, ArrayList<Generator> generators) {
        this.ds = ds;
        this.generators = new ArrayList<>(generators);
    }
}
