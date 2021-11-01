package database;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.NoSuchElementException;
import java.util.Scanner;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.DateTools;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class LuceneBuilder {

    public static class UserData {
        String username;
        String name;
        long userId;
        String userLocation;
        String timezone;
        String language;

        public UserData(String[] fields) {
            username = get(fields[0]);
            name = get(fields[1]);
            userId = Long.parseLong(fields[2]);
            userLocation = get(fields[3]);
            timezone = get(fields[4]);
            language = get(fields[5]);
        }
    }

    public static class TweetData {
        String postTime;
        String postTimezone;

        public TweetData(String[] fields) {
            long time_ms = Long.parseLong(fields[0].split("\\.", 2)[0]) * 1000;
            postTime = DateTools.timeToString(time_ms,
                    DateTools.Resolution.SECOND);
            postTimezone = get(fields[1]);
        }
    }

    private static String get(String cell) {
        if (cell.equals("None") || cell.length() == 0) {
            return "";
        } else {
            return cell;
        }
    }

    public static void addTweet(IndexWriter writer, UserData userData,
            TweetData tweetData, String tweetText) throws IOException {
        Document tweet = new Document();
        // User metadata
        tweet.add(new StringField("username", userData.username,
                Field.Store.YES));
        tweet.add(new StringField("name", userData.name, Field.Store.YES));
        tweet.add(new LongPoint("userId", userData.userId));
        tweet.add(new StringField("userLocation", userData.userLocation,
                Field.Store.YES));
        tweet.add(new StringField("timezone", userData.timezone,
                Field.Store.YES));
        tweet.add(new StringField("language", userData.language,
                Field.Store.YES));

        // Tweet metadata
        tweet.add(new StringField("postTime", tweetData.postTime,
                Field.Store.YES));
        tweet.add(new StringField("postTimezone", tweetData.postTimezone,
                Field.Store.YES));

        // Tweet
        tweet.add(new TextField("text", tweetText, Field.Store.YES));

        writer.addDocument(tweet);
    }

    public static void main(String[] args) throws IOException {
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(index, config);
        Scanner input = new Scanner(System.in);
        while (true) {
            try {
                String[] userFields = input.nextLine().trim().split("\t");
                if (userFields.length == 0) {
                    break;
                }
                UserData userData = new UserData(userFields);

                String[] tweetFields = input.nextLine().trim().split("\t");
                if (tweetFields.length == 0) {
                    break;
                }
                TweetData tweetData = new TweetData(tweetFields);

                String tweet = input.nextLine().trim();

                addTweet(writer, userData, tweetData, tweet);
            } catch (NoSuchElementException e) {
                System.out.println("Done.");
                writer.close();
                input.close();
                return;
            }
        }

        System.out.println("Done.");
        writer.close();
        input.close();
    }
}
