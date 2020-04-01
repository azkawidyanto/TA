package helper;


import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class JsonReaderHelper {
    public static HashMap<Integer, String> readPredicateDictionary(String filename) {
        HashMap<Integer, String> predicateDictionary = new HashMap<>();
        Gson gson = new Gson();
        File file = Paths.get(filename).toFile();

        try {
            JsonObject jsonObject = gson.fromJson(new FileReader(file), JsonObject.class);

            for (Map.Entry<String, JsonElement> entry: jsonObject.entrySet()) {
                predicateDictionary.put(entry.getValue().getAsInt(), entry.getKey());
            }

        } catch(FileNotFoundException f) {
            System.out.println(f.toString());
        }

        return predicateDictionary;
    };
}
