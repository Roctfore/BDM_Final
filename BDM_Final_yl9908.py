
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import trunc
from haversine import haversine
from pyproj import Proj, transform
import sys

def load_and_process_data(spark, output_dir):
    # Load data
    supermarket = spark.read.csv('nyc_supermarkets.csv', header=True, inferSchema=True)
    weekly_patterns = spark.read.csv('/shared/CUSP-GX-6002/data/weekly-patterns-nyc-2019-2020/part-*', header=True, inferSchema=True)

    # 1. Use nyc_supermarkets.csv to filter the visits in the weekly patterns data ( safegraph_placekey column which matches the placekey column in the Weekly Pattern dataset)
    weekly_patterns_filter = weekly_patterns.join(supermarket, weekly_patterns['placekey'] == supermarket['safegraph_placekey'])

    # 2. Only visit patterns with date_range_start or date_range_end overlaps with the 4 months of interests (Mar 2019, Oct 2019, Mar 2020, Oct 2020) will be considered, i.e. either the start or the end date falls within the period.

    # Define the four months of interest
    months_of_interest = ['2019-03-01', '2019-10-01', '2020-03-01', '2020-10-01']

    # Convert date columns to date type
    weekly_patterns_filter = weekly_patterns_filter.withColumn("date_range_start", to_date(col("date_range_start")))
    weekly_patterns_filter = weekly_patterns_filter.withColumn("date_range_end", to_date(col("date_range_end")))

    # Filter DataFrame to include only visit patterns that occur in the months of interest in both date_range_start and date_range_end
    filtered_df = weekly_patterns_filter.filter(
        (trunc(col('date_range_start'), 'MM').isin(months_of_interest)) |
        (trunc(col('date_range_end'), 'MM').isin(months_of_interest))
    )

        # Load the nyc_cbg_centroids file as a Spark DataFrame
    nyc_cbg_centroids = spark.read.csv('nyc_cbg_centroids.csv', header=True, inferSchema=True)

    # 3. Filter CBG FIPS for NYC |Manhattan|36061...||Bronx|36005...|Brooklyn|36047...||Queens|36081...||Staten Island|36085
    nyc_cbg_centroids = nyc_cbg_centroids.filter(
        (nyc_cbg_centroids['cbg_fips'].startswith('36061')) |
        (nyc_cbg_centroids['cbg_fips'].startswith('36005')) |
        (nyc_cbg_centroids['cbg_fips'].startswith('36047')) |
        (nyc_cbg_centroids['cbg_fips'].startswith('36081')) |
        (nyc_cbg_centroids['cbg_fips'].startswith('36085'))
    )

    # Rename the 'latitude' and 'longitude' columns in nyc_cbg_centroids
    nyc_cbg_centroids = nyc_cbg_centroids.withColumnRenamed('latitude', 'latitude_centroids') \
                                        .withColumnRenamed('longitude', 'longitude_centroids')
    
        # Clean the visitor_home_cbgs column in filtered_df to extract CBG FIPS codes
    cleaned_df = filtered_df.withColumn('visitor_home_cbgs', regexp_extract('visitor_home_cbgs', '\\"(\d{12})\\"', 1))

    # Keep the rows in cleaned_df where CBG FIPS column matches with nyc_cbg_centroids
    filtered_df_cbg = cleaned_df.join(nyc_cbg_centroids, cleaned_df['visitor_home_cbgs'] == nyc_cbg_centroids['cbg_fips'], 'inner')

    # Define the conversion function
    def convert_coordinates(lat, lon):
        in_proj = Proj(proj='latlong', datum='WGS84')
        out_proj = Proj(init='epsg:2263', preserve_units=True)
        x, y = transform(in_proj, out_proj, lon, lat)
        return x, y

    # Define a UDF to convert the latitude and longitude columns
    convert_coordinates_udf = udf(convert_coordinates, StructType([StructField('x', DoubleType(), True), StructField('y', DoubleType(), True)]))

    # Convert the latitude and longitude columns in both DataFrames to EPSG 2263
    filtered_df_cbg = filtered_df_cbg.withColumn('visitor_home_cbgs_converted', convert_coordinates_udf(col('latitude_centroids'), col('longitude_centroids')))
    filtered_df_cbg = filtered_df_cbg.withColumn('poi_cbg_converted', convert_coordinates_udf(col('latitude'), col('longitude')))

    # Define the haversine distance function
    def haversine_distance(lat1, lon1, lat2, lon2):
        return haversine((lat1, lon1), (lat2, lon2)) * 0.621371  # Convert kilometers to miles

    # Define a UDF to compute the haversine distance
    haversine_distance_udf = udf(haversine_distance, DoubleType())

    # Compute the haversine distance between visitor_home_cbgs and poi_cbg
    filtered_df_cbg = filtered_df_cbg.withColumn('haversine_distance', haversine_distance_udf(col('latitude_centroids'), col('longitude_centroids'), col('latitude'), col('longitude')))

    # Extract the month and year from date_range_start and date_range_end
    filtered_df_cbg = filtered_df_cbg.withColumn("month_year_start", date_format(col("date_range_start"), "yyyy-MM"))
    filtered_df_cbg = filtered_df_cbg.withColumn("month_year_end", date_format(col("date_range_end"), "yyyy-MM"))

    # Calculate the total distance traveled
    filtered_df_cbg = filtered_df_cbg.withColumn("total_distance", col("haversine_distance") * col("raw_visitor_counts"))

    # Group the data by visitor_home_cbgs and the extracted month-year
    grouped_data = filtered_df_cbg.groupBy("visitor_home_cbgs", "month_year_start", "month_year_end").agg(
        sum("total_distance").alias("sum_total_distance"),
        sum("raw_visitor_counts").alias("sum_visitor_counts")
    )

    # Calculate the average distance
    grouped_data = grouped_data.withColumn("average_distance", col("sum_total_distance") / col("sum_visitor_counts"))

    pivot_df = grouped_data.groupBy("visitor_home_cbgs").pivot("month_year_start").agg(first("average_distance"))
    pivot_df = pivot_df.select('visitor_home_cbgs', '2019-03', '2019-10', '2020-03', '2020-10')

    # Save the DataFrame to the output directory
    pivot_df.write.csv(output_dir)

def main():
    output_dir = sys.argv[1]
    spark = SparkSession.builder \
        .appName("BDM_Final_Challenge_yl9908") \
        .getOrCreate()

    load_and_process_data(spark, output_dir)

if __name__ == "__main__":
    main()