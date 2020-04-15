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
    public static HashMap<Integer, String> readPredicateDictionary(final String filename) {
        final HashMap<Integer, String> predicateDictionary = new HashMap<>();
        final Gson gson = new Gson();
        final File file = Paths.get(filename).toFile();

        try {
            final JsonObject jsonObject = gson.fromJson(new FileReader(file), JsonObject.class);

            for (final Map.Entry<String, JsonElement> entry: jsonObject.entrySet()) {
                predicateDictionary.put(entry.getValue().getAsInt(), entry.getKey());
            }

        } catch(final FileNotFoundException f) {
            System.out.println(f.toString());
        }

        return predicateDictionary;
    };
}
