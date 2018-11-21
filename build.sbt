name := "7200_project"

version := "1.0"

scalaVersion := "2.11.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.0"

libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.7.2"

// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-core
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta2"

// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-nn
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nn" % "1.0.0-beta2"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta2"

libraryDependencies += "com.twelvemonkeys.imageio" % "imageio-core" % "3.3.2"

