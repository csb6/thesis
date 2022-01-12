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

    public static class DocFrequencyCollector extends SimpleCollector {
        public IndexReader reader;
        public Map<String, Long> termToDocFreq = new HashMap<>(1000);
        public Map<String, Long> termToTermFreq = new HashMap<>(1000);
        public int totalTweetsInPeriod = 0;

        public DocFrequencyCollector(IndexReader reader) {
            this.reader = reader;
        }

        @Override
        public void collect(int doc) throws IOException {
            Terms terms = reader.getTermVector(doc, "text");
            ++totalTweetsInPeriod;
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
            Map<String, Double> termToPMI, int totalTweetsInPeriod)
            throws IOException {
        FileWriter outputFile = new FileWriter(filename);
        for (Entry<String, Long> entry : termToDocFreq.entrySet()) {
            String term = entry.getKey();
            long tfPlain = termToTermFreq.get(term);
            if (tfPlain <= 5) {
                continue;
            }
            double tfLog = 1 + Math.log10(tfPlain);
            long df = entry.getValue();
            double idfPlain = (double) totalTweetsInPeriod / (double) df;
            double idfLog = Math.log10(idfPlain);
            double pmi = termToPMI.get(term);
            outputFile.write(term + "\t" + tfPlain + "\t" + tfLog + "\t" + df
                    + "\t" + idfPlain + "\t" + idfLog + "\t"
                    + tfPlain * idfPlain + "\t" + tfPlain * idfLog + "\t"
                    + tfLog * idfPlain + "\t" + tfLog * idfLog + "\t" + pmi);
        }
        outputFile.close();
    }

    public static DocFrequencyCollector collectPeriod(Analyzer analyzer,
            Directory index, String periodQueryText)
            throws ParseException, IOException {
        Query periodQuery = new QueryParser("postTime", analyzer)
                .parse(periodQueryText);

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        DocFrequencyCollector collector = new DocFrequencyCollector(reader);
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

    public static Map<String, Double> collectPMI(DocFrequencyCollector period,
            Map<String, Long> allPeriods) {
        Map<String, Double> result = new HashMap<>(period.termToDocFreq.size());
        long tweetsInPeriod = period.totalTweetsInPeriod;
        for (Entry<String, Long> entry : period.termToDocFreq.entrySet()) {
            String term = entry.getKey();
            long selectedTweetsInPeriod = entry.getValue();
            long totalSelectedTweets = allPeriods.get(term);

            result.put(term, getPMI(selectedTweetsInPeriod, totalSelectedTweets,
                    tweetsInPeriod));
        }
        return result;
    }

    public static void main(String[] args) throws IOException, ParseException {
        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        String period1QueryText = "[2018-03-01T00:00:00 TO 2019-09-01T23:59:59]";
        String period2QueryText = "[2020-03-01T00:00:00 TO 2021-09-01T23:59:59]";

        System.out.println("Collecting data for " + period1QueryText);
        DocFrequencyCollector period1Collector = collectPeriod(analyzer, index,
                period1QueryText);
        System.out.println("Collecting data for " + period2QueryText);
        DocFrequencyCollector period2Collector = collectPeriod(analyzer, index,
                period2QueryText);

        System.out.println("Collecting data for all periods");
        Map<String, Long> allPeriodsDocFrequency = collectTotal(
                period1Collector.termToDocFreq, period2Collector.termToDocFreq);

        System.out.println("Collecting PMI for " + period1QueryText);
        Map<String, Double> period1PMI = collectPMI(period1Collector,
                allPeriodsDocFrequency);
        System.out.println("Collecting PMI for " + period2QueryText);
        Map<String, Double> period2PMI = collectPMI(period2Collector,
                allPeriodsDocFrequency);

        System.out.println("Writing results to file");
        printTermStats("tf-idf-2018-3-1-thru-2019-9-1.txt",
                period1Collector.termToDocFreq, period1Collector.termToTermFreq,
                period1PMI, period1Collector.totalTweetsInPeriod);
        printTermStats("tf-idf-2020-3-1-thru-2021-9-1.txt",
                period2Collector.termToDocFreq, period2Collector.termToTermFreq,
                period2PMI, period2Collector.totalTweetsInPeriod);
    }

}
