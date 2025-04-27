package com.wine_prediction_training.app;

// /**
//  * Hello world!
//  */
// public class App {
//     public static void main(String[] args) {
//         System.out.println("Hello World!");
//     }
// }
// import java.util.logging.LogManager;

// import com.amazonaws.auth.AWSCredentials;
// import com.amazonaws.auth.EnvironmentVariableCredentialsProvider;
// import org.apache.log4j.Level;
// import org.apache.log4j.LogManager;
// import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {
        public static final Logger logger = LoggerFactory.getLogger(App.class);

        // private static final String ACCESS_KEY_ID =  "ASIAQMMMF6R6PDYYYMJE"; //System.getProperty("ACCESS_KEY_ID")
        // private static final String SECRET_KEY = "AJvVXEcpgM11TeuXkaRX6xHVnS+NZ8q3Cnu1LLuK";//System.getProperty("SECRET_KEY")

        // private static final String MASTER_URI = "local[*]";
        private static final String MASTER_URI = "yarn";


        public static void main(String[] args) {

                // SparkSession spark = SparkSession.builder()
                //                 .appName("Wine Quality Prediction App").master(MASTER_URI)
                //                 .config("spark.executor.memory", "3g")
                //                 .config("spark.driver.memory", "3g")
                //                 .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
                //                 .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
                //                 .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                //                 .getOrCreate();
                SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction App")
                .master(MASTER_URI)
                .config("spark.executor.memory", "3g")
                .config("spark.driver.memory", "3g")
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2,org.apache.hadoop:hadoop-common:3.2.2")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .getOrCreate();


                // spark.sparkContext().hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
                //                 "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain");
                

                // spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
                // spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);

                LogisticRegressionV2 parser = new LogisticRegressionV2();
                parser.trainModel(spark);

                spark.stop();
        }
}