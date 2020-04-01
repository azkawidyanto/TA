package data_structure;

public class Predicate {
    public int id;
    public int plusSupport;
    public int negativeSupport;

    public Predicate(int id) {
        this.id = id;
        this.plusSupport = -1;
        this.negativeSupport = -1;
    }

    public boolean isHasSupport() {
        return this.plusSupport != -1 && this.negativeSupport != -1;
    }

    public Predicate(int id, int plusSupport, int negativeSupport) {
        this.id = id;
        this.plusSupport = plusSupport;
        this.negativeSupport = negativeSupport;
    }

    @Override
    public String toString() {
        return String.format("(%d,[%d, %d])", this.id, this.plusSupport, this.negativeSupport);
    }
}
