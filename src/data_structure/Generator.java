package data_structure;

import java.util.ArrayList;

public class Generator {
    public ArrayList<Integer> generator;

    public Generator() {
        generator = new ArrayList<>();
    }

    public Generator(ArrayList<Integer> arrInput) {
        generator = new ArrayList<>(arrInput);
    }

    public Generator(int[] arrInput) {
        generator = new ArrayList<>();
        for (int val: arrInput) {
            generator.add(val);
        }
    }

    public void print() {
        System.out.print("{ ");
        for (Integer val: generator) {
            System.out.printf("%d ", val);
        }
        System.out.print("}");
    }

    @Override
    public String toString() {
        return generator.toString();
    }
}
