package CodeProcessor;

public class Main {
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
                Integer stringIntegerValue = Integer.valueOf(splitedString[splitedString.length - 1]);x
                Support support = new Support(stringIntegerValue);
                dataClasses.add(support);  
            }
        } 
        MemoryWatcher.getInstance().ping();
    }

    public static void main( String[] args) {
        String[] inputs = FileReader.read(filePath);
        v  


    }

    
}