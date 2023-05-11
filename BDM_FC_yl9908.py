import pyspark
import sys
from datetime import datetime
from pyspark import SparkConf, SparkContext

# Set up Spark configuration and context
conf = SparkConf().setAppName("BDM_FC_yl9908")
sc = pyspark.SparkContext.getOrCreate()
output_folder = sys.argv[1]

# Load data
supermarket = sc.textFile('/shared/CUSP-GX-6002/FC/yl9908/nyc_cbg_centroids.csv')
weekly_patterns = sc.textFile('/shared/CUSP-GX-6002/data/weekly-patterns-nyc-2019-2020/*')
nyc_cbg_centroids = sc.textFile('/shared/CUSP-GX-6002/FC/yl9908/nyc_cbg_centroids.csv')

# Convert to RDD
def parse_csv(line):
  return tuple(line.strip().split(','))

supermarket_header = supermarket.first()
supermarket = supermarket.filter(lambda x: x != supermarket_header).map(parse_csv)
weekly_patterns_header = weekly_patterns.first()
weekly_patterns = weekly_patterns.filter(lambda x: x != weekly_patterns_header).map(parse_csv)
nyc_cbg_centroids_header = nyc_cbg_centroids.first()
nyc_cbg_centroids = nyc_cbg_centroids.filter(lambda x: x != nyc_cbg_centroids_header).map(parse_csv)

# Filter weekly_patterns by placekey
supermarket_placekeys = supermarket.map(lambda x: x[0]).collect()
weekly_patterns_filtered = weekly_patterns.filter(lambda x: x[0] in supermarket_placekeys)

# Filter by date_range_start and date_range_end
months_of_interest = ['2019-03-01', '2019-10-01', '2020-03-01', '2020-10-01']
weekly_patterns_filtered = weekly_patterns_filtered.filter(lambda x: x[1][:7] in months_of_interest or x[2][:7] in months_of_interest)

# Extract CBG FIPS codes
def extract_cbg_fips(visitor_home_cbgs_str):
  return visitor_home_cbgs_str.strip().split(':')[0][1:]

weekly_patterns_cbg_fips = weekly_patterns_filtered.map(lambda x: (x[0], x[1], x[2], extract_cbg_fips(x[16])))

# Filter nyc_cbg_centroids
def nyc_cbg_filter(cbg_fips):
  cbg_fips = cbg_fips[0]
  return cbg_fips.startswith('36061') or cbg_fips.startswith('36005') or cbg_fips.startswith('36047') or cbg_fips.startswith('36081') or cbg_fips.startswith('36085')

nyc_cbg_centroids_filtered = nyc_cbg_centroids.filter(nyc_cbg_filter)

# Join weekly_patterns_cbg_fips with nyc_cbg_centroids_filtered
joined_data = weekly_patterns_cbg_fips.map(lambda x: (x[2], x)).join(nyc_cbg_centroids_filtered.map(lambda x: (x[0], x)))

# Define haversine distance function
def distance(lat1, lon1, lat2, lon2):
  # Calculate the difference in latitudes and longitudes
  delta_lat = lat2 - lat1
  delta_lon = lon2 - lon1

  # Calculate the great circle distance between the two points using the Haversine formula
  a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
  c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
  distance = 3961*c # Radius of Earth in miles
  return distance

# Calculate haversine distance for each row

def compute_distance(row):
  _, ((_, date_range_start, date_range_end, visitor_home_cbgs), (_, lat1, lon1)) = row
  lat2, lon2 = float(lat1), float(lon1)
  lat1, lon1 = float(row[1][0][3]), float(row[1][0][4])
  distance = distance(lat1, lon1, lat2, lon2)
  return visitor_home_cbgs, date_range_start[:7], date_range_end[:7], distance, int(row[1][0][14])

# Compute distance for each row and filter out rows with zero raw_visitor_counts
distances = joined_data.map(compute_distance).filter(lambda x: x[4] > 0)

# Calculate total and average distance
def compute_totals(row):
  visitor_home_cbgs, date_range_start, date_range_end, distance, visitor_count = row
  return (visitor_home_cbgs, date_range_start, date_range_end), (distance * visitor_count, visitor_count)

totals = distances.map(compute_totals).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

# Calculate average distance
averages = totals.map(lambda x: (x[0], x[1][0] / x[1][1]))

# Pivot the data
def pivot_key(row):
  visitor_home_cbgs, date_range_start, date_range_end = row[0]
  return visitor_home_cbgs, (date_range_start, row[1])

pivoted_data = averages.map(pivot_key).groupByKey().mapValues(dict)

# Save output
def format_output(row):
  return f"{row[0]},{row[1].get('2019-03', '')},{row[1].get('2019-10', '')},{row[1].get('2020-03', '')},{row[1].get('2020-10', '')}"

output = pivoted_data.map(format_output)
output.saveAsTextFile(output_folder)
