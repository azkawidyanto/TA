package helper;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

import constants.CommonConstant;

public class FileReader {
    public static String[] read(String filepath) {
        ArrayList<String> result;

        ArrayList<ArrayList<String>>  records = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filepath))) {
            while (scanner.hasNextLine()) {
                records.add(getRecordFromLine(scanner.nextLine()));
            }
        } catch (Exception e) {
            System.out.println("Error while reading file: " + e.toString());
        }

        result = convertToStringList(records);

        return result.toArray(new String[0]);
    }

    public static Integer getNumberOfPredicate(String filepath) {
        Integer result = 0;
        try (Scanner scanner = new Scanner(new File(filepath))) {
            if (scanner.hasNextLine()) {
                result = getNumberOfPredicateFromLine(scanner.nextLine());
            }
        } catch (Exception e) {
            System.out.println("Error while reading file: " + e.toString());
        }

        return result;
    }

    public static Integer getNumberOfPredicateFromLine(String line) {
        Integer result = 0;
        try (Scanner rowScanner = new Scanner(line)) {
            rowScanner.useDelimiter(CommonConstant.COMMA_DELIMITER);
            while (rowScanner.hasNext()) {
                result += 1;
                rowScanner.next();
            }
        } catch (Exception e) {
            System.out.println("Error while reading line: " + e.toString());
        }
        return Math.max(0, result - 1);
    }


    public static String[] readBugPredicates(String filepath) {
        ArrayList<ArrayList<String>>  records = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filepath))) {
            while (scanner.hasNextLine()) {
                records.add(getRecordFromLine(scanner.nextLine()));
            }
        } catch (Exception e) {
            System.out.println("Error while reading file: " + e.toString());
        }

        return records.get(0).toArray(new String[0]);
    }

    private static ArrayList<String> convertToStringList(ArrayList<ArrayList<String>> records) {
        ArrayList<String> compressedData = new ArrayList<>();
        for (ArrayList<String> line: records) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < line.size() - 1; i++) {
                if (Integer.valueOf(line.get(i)) == 1) {
                    sb.append(i+1);
                    sb.append(" ");
                }
            }
            sb.append(line.get(line.size() - 1));
            compressedData.add(sb.toString());
        }

        return compressedData;
    }

    private static ArrayList<String> getRecordFromLine(String line) {
        ArrayList<String> values = new ArrayList<>();
        try (Scanner rowScanner = new Scanner(line)) {
            rowScanner.useDelimiter(CommonConstant.COMMA_DELIMITER);
            while (rowScanner.hasNext()) {
                values.add(rowScanner.next());
            }
        } catch (Exception e) {
            System.out.println("Error while reading line: " + e.toString());
        }

        return values;
    }
}
