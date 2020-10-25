import java.io.FileReader;

public class test {
    public static void main(String[] arg) {
        String str = FileReader.read("test.txt");

        String split[] = str.split("\n");
        for (String s: split)
            System.out.println(s);
    }
}
