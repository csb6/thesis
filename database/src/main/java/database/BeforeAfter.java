package database;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
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

    public static class DocFrequencyCollector extends SimpleCollector {
        public IndexReader reader;
        public Map<String, Long> termToDocFreq = new HashMap<>(1000);
        public Map<String, Long> termToTermFreq = new HashMap<>(1000);
        public int hitCount = 0;

        public DocFrequencyCollector(IndexReader reader) {
            this.reader = reader;
        }

        @Override
        public void collect(int doc) throws IOException {
            Terms terms = reader.getTermVector(doc, "text");
            ++hitCount;
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

    public static void printTFByIDF(Map<String, Long> termToDocFreq,
            Map<String, Long> termToTermFreq, int hitCount) {
        for (Entry<String, Long> entry : termToDocFreq.entrySet()) {
            String term = entry.getKey();
            long tfPlain = termToTermFreq.get(term);
            double tfLog = 1 + Math.log10(tfPlain);
            long df = entry.getValue();
            double idfPlain = (double)hitCount / (double)df;
            double idfLog = Math.log10(idfPlain);
            System.out.println(term + "\t" + tfPlain + "\t" + tfLog  + "\t"
                               + df + "\t"
                               + idfPlain + "\t" + idfLog + "\t"
                               + tfPlain * idfPlain + "\t" + tfPlain * idfLog + "\t"
                               + tfLog * idfPlain + "\t" + tfLog * idfLog);
        }
    }

    public static void main(String[] args) throws IOException, ParseException {
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        if (args.length == 0) {
            System.err.println("Error: pass in query as an argument");
            analyzer.close();
            return;
        }
        String queryText = args[0];

        Query query = new QueryParser("postTime", analyzer).parse(queryText);

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        DocFrequencyCollector collector = new DocFrequencyCollector(reader);

        searcher.search(query, collector);

        System.out.println("Results for query: '" + queryText + "'");
        printTFByIDF(collector.termToDocFreq,
                collector.termToTermFreq, collector.hitCount);
    }

}
