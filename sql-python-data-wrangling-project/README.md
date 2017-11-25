# Intro

In this project, I clean the Houston portion of the [OpenStreetMap](https://www.openstreetmap.org) data by using data munging techniques, such as assessing the quality of the data for validity, accuracy, completeness, consistency and uniformity. After cleaning the data, I explore the Houston street map data using sql queries.

First, I import the xml data and clean street type and zip code inconsistencies before exporting the data into csvs, which I use to create the houston.db using sql. Finally, I take a granular look at the data by running sql queries to find insight about the data.

# Included Files

1. audit_postcode.py:
	The code I used to audit zip codes

2. audit_postcode_fix.py:
	The code I used to fix the zip codes

3. audit_street_type.py:
	The code I used to audit street types

4. audit_street_type_fix.py:
	The code I used to fix street type abbreviations

5. export.py:
	The code I used to export and clean the OSM file to CSV

6. sample.osm:
	Sample of the Houston dataset

7. schema.py:
 The schema for sqlite Houston database

8. Write Up.pdf:
	The actual project write up portion
