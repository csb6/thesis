package database;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.NoSuchElementException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.DateTools;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
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

    public static void addTweet(IndexWriter writer,
            FieldType textWithTermVectorType, UserData userData,
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
        tweet.add(new Field("text", tweetText, textWithTermVectorType));

        writer.addDocument(tweet);
    }

    public static void main(String[] args) throws IOException {
        FieldType textWithTermVectorType = new FieldType(TextField.TYPE_STORED);
        textWithTermVectorType.setStoreTermVectors(true);

        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = FSDirectory.open(Paths.get("index"));

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(index, config);
        BufferedReader input = new BufferedReader(
                new InputStreamReader(System.in, "UTF-8"));
        long lineNum = 0;
        while (true) {
            try {
                ++lineNum;
                String[] userFields = input.readLine().trim().split("\t");
                if (userFields.length == 0) {
                    System.out.println(
                            "Line " + lineNum + " was empty, stopped there");
                    break;
                }
                UserData userData = new UserData(userFields);

                ++lineNum;
                String[] tweetFields = input.readLine().trim().split("\t");
                if (tweetFields.length == 0) {
                    System.out.println(
                            "Line " + lineNum + " was empty, stopped there");
                    break;
                }
                TweetData tweetData = new TweetData(tweetFields);

                ++lineNum;
                String tweet = input.readLine().trim();

                addTweet(writer, textWithTermVectorType, userData, tweetData,
                        tweet);
            } catch (NoSuchElementException e) {
                System.out.println("Done.");
                writer.close();
                input.close();
                return;
            } catch (IOException e) {
                System.out
                        .println("Error during processing of line " + lineNum);
                e.printStackTrace();
                writer.close();
                input.close();
                return;
            } catch (Exception e) {
                System.out.println("Error during processing of line " + lineNum);
                e.printStackTrace();
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
