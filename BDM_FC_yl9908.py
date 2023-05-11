import csv
import json
import numpy as np
import math
import pandas as pd
import datetime
import pyproj
import IPython

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()

WEEKLY_PATTERN = '/shared/CUSP-GX-6002/data/weekly-patterns-nyc-2019-2020/part-*'
NYC_SUPERMARKETS = os.getenv('SPARK_YARN_STAGING_DIR')+'/nyc_supermarkets.csv'
NYC_CBG_CENTROIDS = os.getenv('SPARK_YARN_STAGING_DIR')+'/nyc_cbg_centroids.csv'

nyc_supermarkets = sc.textFile(NYC_SUPERMARKETS).filter(lambda x: not x.startswith('place_id,latitude,'))\
            .map(lambda line: next(csv.reader([line])))
safegraph_placekeys = set(nyc_supermarkets.map(lambda x: x[-2]).collect())


proj = pyproj.Proj(init='EPSG:2263', preserve_units=True)

nyc_cbg_centroids = sc.textFile(NYC_CBG_CENTROIDS) \
                     .filter(lambda x: not x.startswith('cbg_fips,')) \
                     .map(lambda line: next(csv.reader([line])))\
                     .filter(lambda x: any(x[0].startswith(prefix) for prefix in ['36061','36005','36047','36081','36085']))\
                     .map(lambda x: (x[0], proj(float(x[2]), float(x[1]))))
cbgs_geodict = nyc_cbg_centroids.collectAsMap()
nyc_cbg_fips = set(cbgs_geodict.keys())


start_dates = [datetime.date(2019, 3, 1), datetime.date(2019, 10, 1), datetime.date(2020, 3, 1), datetime.date(2020, 10, 1)]
end_dates = [datetime.date(2019, 4, 1), datetime.date(2019, 11, 1), datetime.date(2020, 4, 1), datetime.date(2020, 11, 1)]
date_label = ['2019-03', '2019-10', '2020-03', '2020-10']

def parse_date(datestr):
    return datetime.datetime.strptime(datestr[:10], '%Y-%m-%d').date()
def filter_by_date_range(record):
    start_date = parse_date(record[12])
    end_date = parse_date(record[13])
    for i in range(4):
        if start_dates[i] <= start_date < end_dates[i] or start_dates[i] <= end_date < end_dates[i]:
            return True
    return False
def label_date_range(start_date, end_date):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    # label date with: 2019-03	2019-10	2020-03	2020-10
    for i in range(4):
        if start_dates[i] <= start_date < end_dates[i] or start_dates[i] <= end_date < end_dates[i]:
            return date_label[i]
    return 'None'

def filter_cbgs(home_cbgs_dict, nyc_cbg_fips):
  # first filter by borough to reduce the number of times to traverse the fips list
  borough_keys = [k for k in home_cbgs_dict.keys() if k[:5] in ['36061', '36005', '36047', '36081', '36085']]
  # then filter by fips in nyc_cbg_centroids
  fips_keys = [k for k in borough_keys if k in nyc_cbg_fips]

  filtered_dict = {k: home_cbgs_dict[k] for k in fips_keys}

  return filtered_dict


def distance(coordinate1,coordinate2):
  x1, y1 = coordinate1
  x2, y2 = coordinate2
  return 0.000621371192*((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_distance_for_each_month(values):
  value_list = ['','','','']
  month = ['2019-03', '2019-10', '2020-03', '2020-10']
  for record in values:
    for i in range(4):
      if record[0] == month[i]:
        value_list[i] = record[1]
  return tuple(value_list)

weekly_pattern = sc.textFile(WEEKLY_PATTERN).filter(lambda x: not x.startswith('"placekey","'))\
            .map(lambda line: next(csv.reader([line])))\
            .filter(lambda x: x[0] in safegraph_placekeys)\
            .filter(filter_by_date_range)\
            .map(lambda x: (x[18], filter_cbgs(json.loads(x[19]), nyc_cbg_fips), label_date_range(x[12],x[13])))\
            .filter(lambda x: not x[1]=={})\
            .map(lambda x: ({(k,x[2]): (distance(cbgs_geodict[x[0]],cbgs_geodict[k]),x[1][k]) for k in x[1].keys()}))\
            .flatMap(lambda x: [(k, x[k]) for k in x.keys()])\
            .groupByKey()\
            .mapValues(lambda values: sum([distance * weight for distance, weight in values]) / sum([weight for distance, weight in values]))\
            .map(lambda x: (x[0][0], (x[0][1], x[1])))\
            .groupByKey()\
            .mapValues(lambda values: get_distance_for_each_month(values))\
            .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3]))\
            .sortBy(lambda x: x[0])

header = 'cbg_fips,2019-03,2019-10,2020-03,2020-10'
weekly_pattern.coalesce(1)\
    .mapPartitions(lambda x: itertools.chain([header], x)) \
    .saveAsTextFile(sys.argv[1] if len(sys.argv)>1 else 'outputfc')
