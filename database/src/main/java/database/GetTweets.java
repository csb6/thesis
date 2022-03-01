package database;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class GetTweets {

    public static void main(String[] args)
            throws IOException, ParseException, java.text.ParseException {
        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        if (args.length != 2) {
            System.err.println("Usage: ./GetTweets [startDate] [endDate]");
            analyzer.close();
            return;
        }
        if (args[0].length() != 8 || args[1].length() != 8) {
            System.err.println("Error: date must be in form: YYYYMMDD");
            analyzer.close();
            return;
        }
        String queryText = String.format("[%s000000 TO %s999999]", args[0],
                args[1]);

        Query query = new QueryParser("postTime", analyzer).parse(queryText);
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        TotalHitCountCollector collector = new TotalHitCountCollector();
        searcher.search(query, collector);
        if (collector.getTotalHits() == 0) {
            System.err.println("No results for query: " + queryText);
            analyzer.close();
            return;
        }

        TopDocs docs = searcher.search(query, collector.getTotalHits());
        Set<String> fields = new HashSet<>(
                Arrays.asList("userLocation", "timezone", "postTime", "text"));
        for (ScoreDoc match : docs.scoreDocs) {
            Document doc = reader.document(match.doc, fields);
            System.out.println(String.join("\t", doc.get("userLocation"),
                    doc.get("timezone"), doc.get("postTime"), doc.get("text")));
        }
        analyzer.close();
    }

}
