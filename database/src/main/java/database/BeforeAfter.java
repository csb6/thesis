package database;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
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
            Map<String, Double> termToPMIDiff, int totalTweetsInPeriod)
            throws IOException {
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
            double pmiDiff = termToPMIDiff.get(term);
            outputFile.write(term + "\t" + tfPlain + "\t" + tfLog + "\t" + df
                    + "\t" + idfPlain + "\t" + idfLog + "\t"
                    + tfPlain * idfPlain + "\t" + tfPlain * idfLog + "\t"
                    + tfLog * idfPlain + "\t" + tfLog * idfLog + "\t" + pmiDiff
                    + "\n");
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

        Map<String, Double> pmiDiff = collectPMIDiff(period1PMI, period2PMI);

        System.out.println("Writing results to file");
        printTermStats("tf-idf-2018-3-1-thru-2019-9-1.txt",
                period1.termToDocFreq, period1.termToTermFreq, pmiDiff,
                period1.totalTweets);
        printTermStats("tf-idf-2020-3-1-thru-2021-9-1.txt",
                period2.termToDocFreq, period2.termToTermFreq, pmiDiff,
                period2.totalTweets);
    }

}
