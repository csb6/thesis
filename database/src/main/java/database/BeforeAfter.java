package database;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

public class BeforeAfter {

    public static class TermStatsCollector extends SimpleCollector {
        public IndexReader reader;
        // Maps terms -> the number of documents (in this selection) that
        // they appear in
        public Map<String, Long> termToDocFreq = new HashMap<>(1000);
        // Maps terms -> the number of times the term occurs across all
        // documents (in this selection)
        public Map<String, Long> termToTermFreq = new HashMap<>(1000);
        // The number of tweets (documents) in the selection
        public int totalTweets = 0;

        public TermStatsCollector(IndexReader reader) {
            this.reader = reader;
        }

        @Override
        public void collect(int doc) throws IOException {
            Terms terms = reader.getTermVector(doc, "text");
            if (terms == null) {
                return;
            }
            ++totalTweets;
            TermsEnum termsInDoc = terms.iterator();
            BytesRef currTermInDoc = termsInDoc.next();
            while (currTermInDoc != null) {
                String term = currTermInDoc.utf8ToString();
                Long docFreq = termToDocFreq.get(term);
                if (docFreq == null) {
                    termToDocFreq.put(term, 1l);
                } else {
                    termToDocFreq.put(term, docFreq + 1);
                }

                Long termFreq = termToTermFreq.get(term);
                PostingsEnum docInfo = MultiFields.getTermDocsEnum(reader,
                        "text", currTermInDoc);
                if (docInfo.docID() == -1) {
                    docInfo.nextDoc();
                }
                if (termFreq == null) {
                    termToTermFreq.put(term, (long) docInfo.freq());
                } else {
                    termToTermFreq.put(term, termFreq + docInfo.freq());
                }

                currTermInDoc = termsInDoc.next();
            }
        }

        @Override
        public boolean needsScores() {
            return false;
        }
    }

    public static int getHitCount(IndexSearcher searcher, Query query)
            throws IOException {
        TotalHitCountCollector collector = new TotalHitCountCollector();
        searcher.search(query, collector);
        return collector.getTotalHits();
    }

    public static class Pair implements Comparable<Pair> {
        String term;
        float tfByIdf;

        final float Delta = 0.001f;

        public Pair(String term, float tfByIdf) {
            this.term = term;
            this.tfByIdf = tfByIdf;
        }

        @Override
        public int compareTo(Pair other) {
            float difference = other.tfByIdf - tfByIdf;
            if (difference > Delta) {
                return -1;
            } else if (Math.abs(difference) <= Delta) {
                return 0;
            } else {
                return 1;
            }
        }

    }

    public static void printTermStats(String filename,
            Map<String, Long> termToDocFreq, Map<String, Long> termToTermFreq,
            Map<String, Double> termToPMI, Map<String, Double> termToPMIDiff,
            int totalTweetsInPeriod) throws IOException {
        FileWriter outputFile = new FileWriter(filename);
        for (Entry<String, Long> entry : termToDocFreq.entrySet()) {
            String term = entry.getKey();
            long tfPlain = termToTermFreq.get(term);
            if (tfPlain <= 15 || !termToPMIDiff.containsKey(term)) {
                // Ignore rare terms or terms that only appear in one of the
                // periods
                continue;
            }
            double tfLog = 1 + Math.log10(tfPlain);
            long df = entry.getValue();
            double idfPlain = (double) totalTweetsInPeriod / (double) df;
            double idfLog = Math.log10(idfPlain);
            double pmi = termToPMI.get(term);
            double pmiDiff = termToPMIDiff.get(term);
            outputFile.write(term + "\t" + tfPlain + "\t" + tfLog + "\t" + df
                    + "\t" + idfPlain + "\t" + idfLog + "\t"
                    + tfPlain * idfPlain + "\t" + tfPlain * idfLog + "\t"
                    + tfLog * idfPlain + "\t" + tfLog * idfLog + "\t" + pmi
                    + "\t" + pmiDiff + "\n");
        }
        outputFile.close();
    }

    public static TermStatsCollector collectPeriod(Analyzer analyzer,
            Directory index, String periodQueryText)
            throws ParseException, IOException {
        Query periodQuery = new QueryParser("postTime", analyzer)
                .parse(periodQueryText);

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        TermStatsCollector collector = new TermStatsCollector(reader);
        searcher.search(periodQuery, collector);

        return collector;
    }

    public static double getPMI(long selectedTweetsInPeriod,
            long totalSelectedTweets, long tweetsInPeriod) {
        return ((double) selectedTweetsInPeriod)
                / ((double) totalSelectedTweets) * ((double) tweetsInPeriod);
    }

    public static Map<String, Long> collectTotal(Map<String, Long> period1,
            Map<String, Long> period2) {
        Map<String, Long> result = Stream.of(period1, period2)
                .flatMap(map -> map.entrySet().stream())
                .collect(Collectors.toMap(Map.Entry::getKey,
                        Map.Entry::getValue, (v1, v2) -> v1 + v2));
        return result;
    }

    public static Map<String, Double> collectPMI(TermStatsCollector period,
            Map<String, Long> allPeriods) {
        Map<String, Double> result = new HashMap<>(period.termToDocFreq.size());
        for (Entry<String, Long> entry : period.termToDocFreq.entrySet()) {
            String term = entry.getKey();
            long selectedTweetsInPeriod = entry.getValue();
            long totalSelectedTweets = allPeriods.get(term);

            result.put(term, getPMI(selectedTweetsInPeriod, totalSelectedTweets,
                    period.totalTweets));
        }
        return result;
    }

    public static Map<String, Double> collectPMIDiff(
            Map<String, Double> period1PMI, Map<String, Double> period2PMI) {
        Map<String, Double> result = new HashMap<>(2000);
        for (Entry<String, Double> entry : period1PMI.entrySet()) {
            String term = entry.getKey();
            double period1PMIValue = entry.getValue();
            Double period2PMIValue = period2PMI.get(term);
            if (period2PMIValue == null) {
                // Term isn't found in period2; ignore it
                continue;
            }
            result.put(term, period2PMIValue - period1PMIValue);
        }
        return result;
    }

    public static Map<String, Double> collectPMIDiffFoodWords(
            Map<String, Double> period1PMI, Map<String, Double> period2PMI) {
        Map<String, Double> result = new HashMap<>(2000);
        for (Entry<String, Double> entry : period1PMI.entrySet()) {
            String term = entry.getKey();
            double period1PMIValue = entry.getValue();
            Double period2PMIValue = period2PMI.get(term);
            if (period2PMIValue == null) {
                // Term isn't found in period2; ignore it
                continue;
            }
            if (!healthyFoods.contains(term) && !neutralFoods.contains(term)
                    && !unhealthyFoods.contains(term)) {
                continue;
            }
            result.put(term, period2PMIValue - period1PMIValue);
        }
        return result;
    }

    static Set<String> healthyFoods = new HashSet<>(Arrays.asList("pea",
            "coconut", "caviar", "horseradish", "monkfish", "eggplant", "salad",
            "watermelon", "alfalfa", "nut", "mulberry", "apple", "pomegranate",
            "barley", "acorn", "mullet", "kale", "roe", "shellfish",
            "boysenberry", "maize", "fig", "lemongrass", "pepper", "lemon",
            "jackfruit", "shallot", "strawberry", "scallop", "tomatillo",
            "tofu", "watercress", "clove", "sesame", "breast", "courgette",
            "stir fry", "passionfruit", "asparagus", "berry", "peanut",
            "banana", "broccoli", "tomato", "pomelo", "beet", "nectarine",
            "micronutrient", "pilaf", "spinach", "zucchini", "carrot", "bream",
            "persimmon", "kiwi", "diet", "scramble", "quinoa", "seed", "shell",
            "collard", "guava", "puree", "date", "sauerkraut", "hummus",
            "parsnip", "aubergine", "bean", "oyster", "nutrient", "tomatoe",
            "succotash", "honeydew", "beancurd", "pumpernickel", "soy",
            "kumquat", "avacado", "iceberg", "bamboo", "unleavened", "green",
            "leek", "endive", "gherkin", "poppyseed", "wheat", "avocado",
            "elderberry", "lime", "guacamole", "blueberry", "papaya", "rye",
            "dragonfruit", "blanch", "kohlrabi", "pattypan", "rhubarb",
            "salmon", "vege", "clam", "cranberry", "cod", "pickle", "walnut",
            "cantaloupe", "plum", "tangerine", "soybean", "whey", "conger",
            "gooseberry", "daikon", "grape", "greenbean", "ginger", "plantain",
            "legume", "grapefruit", "cabbage", "fruit", "okra", "cauliflower",
            "sorghum", "almond", "grain", "durian", "garlic", "blackberry",
            "scallion", "cherry", "haddock", "prune", "vitamin", "brocolli",
            "orange", "peach", "jicama", "citrus", "quince", "anchovy",
            "cashew", "citron", "yam", "tea", "pear", "bran", "peapod", "melon",
            "millet", "cucumber", "pumpkin", "mushroom", "applesauce", "tuber",
            "celery", "beetroot", "eel", "olive", "pineapple", "black eyed pea",
            "squid", "breadfruit", "mandarin", "ceasar", "raspberry", "fava",
            "subway", "artichoke", "sage", "borage", "marionberry", "shrimp",
            "apricot", "crab", "jalapeno", "onion", "hake", "bass", "radish",
            "chestnut", "tuna", "loquat", "lettuce", "snail", "romaine",
            "sprout", "parsley", "lentil", "lychee", "chard", "prawn", "ceaser",
            "seaweed", "greengage", "hazelnut", "chickpea", "houmous",
            "chalote", "ugli", "lox", "squash", "flax", "groundnut", "crouton",
            "trout", "pecan", "gruel", "turnip"));

    static Set<String> neutralFoods = new HashSet<>(Arrays.asList("protein",
            "curry", "entree", "rib", "herb", "crust", "kidney", "roast",
            "tart", "chutney", "herre", "buckwheat", "mash", "munch", "paprika",
            "ice", "hot", "spud", "mustard", "ketchup", "shish", "goulash",
            "turkey", "dough", "lamb", "spice", "licorice", "corn", "venison",
            "oxtail", "pasta", "clove", "submarine", "cider", "rosemary",
            "bagel", "milky", "seafood", "sole", "omnivore", "halibut",
            "mincemeat", "butter", "soup", "z", "crayfish", "nutmeg", "taro",
            "leave", "mango", "tarragon", "thyme", "tortilla", "chicken",
            "leaf", "curd", "mutton", "soysauce", "sardine", "edible", "food",
            "sorrel", "layer", "teriyaki", "water", "pita", "oatmeal", "stalk",
            "cumin", "dandelion", "dog", "yolk", "quiche", "roll", "omelette",
            "pot", "sirloin", "ration", "fish", "anise", "buttermilk",
            "cornflake", "chicory", "ham", "platter", "cornmeal", "mint",
            "dogfish", "swiss", "chill", "gelatin", "tenderloin", "partridge",
            "foodstuff", "meat", "spareribs", "breadstick", "potato", "milk",
            "goosefish", "lunch", "sweetbread", "cereal", "clam", "french",
            "dip", "sushi", "marrow", "dry", "loaf", "sauce", "garnish",
            "stomach", "stew", "cinnamon", "spicy", "supper", "cracker",
            "saffron", "broth", "chili", "salsa", "grit", "heart", "oil",
            "sandwich", "bake", "coffee", "glaze", "egg", "drumstick",
            "gourmet", "gravy", "offal", "stiff", "refreshment", "bread",
            "dill", "noodle", "marinate", "oat", "popcorn", "sweetcorn",
            "mineral", "salt", "stuff", "wafer", "garden", "chick", "dietician",
            "pop", "quail", "ravioli", "duck", "liver", "savory", "wasabi",
            "fennel", "pretzel", "grub", "currant", "saute", "breakfast",
            "bitter", "peppercorn", "fillet", "coriander", "broil", "meatloaf",
            "nosh", "slaw", "sustenance", "sweet", "breadcrumb", "drink",
            "casserole", "rice", "juicy", "marinade", "veal", "greengrocer",
            "appetizer", "goose", "dress", "mince", "rabbit", "coleslaw",
            "carbohydrate", "oregano", "meatball", "cassava", "mussel", "cocoa",
            "crunch", "bayleaf", "cornflour", "cuisine", "mead", "spaghetti",
            "flour", "yogurt", "ingredient", "non fatten", "turmeric", "season",
            "vanilla", "pork", "omelet", "lobster", "vinegar", "comestible",
            "brunch", "chive", "granola", "raisin", "cilantro", "tapioca",
            "yeast", "hare", "dinner", "gelatine", "steak", "caper", "crouton",
            "basil"));

    static Set<String> unhealthyFoods = new HashSet<>(Arrays.asList("fondue",
            "oleo", "brandy", "cinnabon", "pringle", "rib", "brisket", "kebab",
            "tartar", "tart", "toffee", "coke", "nachos", "brownie", "ice",
            "cannelloni", "gridiron", "blancmange", "mochi", "cocktail", "suet",
            "frost", "freeze", "refrie", "rennet", "sweetener", "latte",
            "crispy", "relish", "jimmy", "koolaid", "cream", "pancake", "bagel",
            "choclate", "burrito", "cheesecake", "macaroni", "lemonade",
            "batter", "cheese", "gingerale", "sourdough", "pie", "whiskey",
            "juice", "pasty", "gatorade", "bourbon", "marshmallow", "gouda",
            "cookie", "budweiser", "grease", "doughnut", "ground", "torte",
            "baguette", "ramen", "cochinillo", "sausage", "brew", "caramel",
            "cupcake", "poptart", "whopper", "hash", "mint", "cappuccino",
            "cornbread", "bone", "icee", "lunchmeat", "canneloni", "beef",
            "flan", "carmel", "quesadilla", "pizza", "soda", "snicker",
            "gristle", "scone", "pastry", "gyro", "fry", "marmalade",
            "pepperoni", "sherry", "salami", "redbull", "bacon", "pate", "bbq",
            "macaroon", "grill", "punch", "chilli", "molasse", "edam",
            "mayonnaise", "shortcake", "sundae", "sugar", "whip", "knucklebone",
            "cracker", "oreo", "barbecue", "whataburger", "tonic", "kreme",
            "cofee", "nectar", "mozzarella", "pudde", "chip", "jam", "coffee",
            "cobbler", "waffle", "liquor", "beer", "cheeseburger",
            "preservative", "blood", "margarine", "donut", "frappuccino",
            "cheddar", "baste", "cake", "barbeque", "dessert", "jello", "crepe",
            "fast", "junket", "syrup", "cheeto", "creme", "pretzel", "danish",
            "croissant", "powerade", "junk", "tamal", "wine", "scotch",
            "butterscotch", "muffin", "toast", "strudel", "hamburger", "tamale",
            "jelly", "crackle", "biscuit", "calzone", "milkshake", "fat",
            "meringue", "snack", "aspic", "candy", "taco", "cheesesteak",
            "sultana", "custard", "wing", "dumple", "shake", "fajita", "burger",
            "fritter", "eggnog", "sherbet", "tripe", "hotdog", "patty",
            "popover", "marshmellow", "lasagna", "kool aid", "popsicle",
            "honey", "lard", "crisp", "vodka", "sub", "chocolate", "frozen",
            "sorbet", "enchilada", "smoke"));

    public static void main(String[] args) throws IOException, ParseException {
        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        String period1QueryText = "[2018-03-01T00:00:00 TO 2019-09-01T23:59:59]";
        String period2QueryText = "[2020-03-01T00:00:00 TO 2021-09-01T23:59:59]";

        System.out.println("Collecting data for " + period1QueryText);
        TermStatsCollector period1 = collectPeriod(analyzer, index,
                period1QueryText);
        System.out.println("Collecting data for " + period2QueryText);
        TermStatsCollector period2 = collectPeriod(analyzer, index,
                period2QueryText);

        System.out.println("Collecting data for all periods");
        Map<String, Long> allPeriodsDocFrequency = collectTotal(
                period1.termToDocFreq, period2.termToDocFreq);

        System.out.println("Collecting PMI for " + period1QueryText);
        Map<String, Double> period1PMI = collectPMI(period1,
                allPeriodsDocFrequency);
        System.out.println("Collecting PMI for " + period2QueryText);
        Map<String, Double> period2PMI = collectPMI(period2,
                allPeriodsDocFrequency);

        // Map<String, Double> pmiDiff = collectPMIDiff(period1PMI, period2PMI);
        Map<String, Double> pmiDiff = collectPMIDiffFoodWords(period1PMI,
                period2PMI);

        System.out.println("Writing results to file");
        printTermStats("tf-idf-pmi-2018-3-1-thru-2019-9-1.txt",
                period1.termToDocFreq, period1.termToTermFreq, period1PMI,
                pmiDiff, period1.totalTweets);
        printTermStats("tf-idf-pmi-2020-3-1-thru-2021-9-1.txt",
                period2.termToDocFreq, period2.termToTermFreq, period2PMI,
                pmiDiff, period2.totalTweets);
    }

}
