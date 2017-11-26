# Project Write Up

### INTRODUCTION
I chose to clean the Houston data of OpenStreetMap because I was born and raised in Houston! You can access the Houston OSM data [here](https://mapzen.com/data/metro-extracts/metro/houston_texas/).

### DATA WRANGLING AND CLEANING
The two main issues with this dataset that I found were street type abbreviations and postal codes(zipcodes). I programmatically found, cleaned, and standardized them in the following process.

#### Street Types

After auditing the street types, I created a mapping of the main street types that I needed to change in the dataset.
```
mapping = { 'Ave': 'Avenue',
			'Ave.': 'Avenue',
			'Blvd': 'Boulevard',
			'Blvd.': 'Bouelvard',
			'Dr': 'Drive',
			'Frwy': 'Freeway',
			'Fwy': 'Freeway',
			'HIGHWAY': 'Highway',
			'Ln': 'Lane',
			'Pkwy': 'Parkway',
			'Rd': 'Road',
			'St': 'Street',
			'Stree': 'Street',
			'street': 'Street'
            }
```

Then, I used this function to clean and standardized the street types:
```
def update_name(name, mapping):
    list_name = name.split(' ')
    last_type = list_name.pop()
    if last_type in mapping:
    	new_type = mapping[last_type]
    	list_name.append(new_type)
    	name = ' '.join(list_name)

    	return name
    else:
    	return name
```

#### Postal Codes

After auditing the postal codes, I found that there were codes such as:```TX77024```with ```'TX'``` in front of the zip code, and ```77384-xxxx``` with ```'-xxxx'``` after the zip code.

In order to clean this these inconsistencies, I wrote a cleaning function that took just the 5 digit zip code and ignored anything before or after it like the ```'TX'``` and
the ```'-xxxx'``` parts:

```
def fix_postcode(code):
	if len(code) >= 5:
		start_postcode = code.find('7')
		end_postcode = start_postcode + 5
		new_code = code[start_postcode:end_postcode]

		return new_code
	else:
		return code
```
I noticed there were also many tags with ```'tiger'``` as attributes. After doing some research, I found that 'tiger' stood for The Topologically
Integrated Geographic Encoding and Referencing system.
 I decided to leave these 'tiger'
attributes how they were since it was a reliable source of data.


### STATISTICAL OVERVIEW OF DATASET

###### 1. SIZE OF FILE
I used the following queries to find the file size of houston.db:
```
PRAGMA page_size;
4096

PRAGMA page_count;
121466

4096 * 121466 / 1,000,000 #divide to convert bytes to MB
Answer: 497.52 MB
```

##### 2. NUMBER OF UNIQUE USERS
```
SELECT COUNT(DISTINCT(e.uid))
FROM
(SELECT uid
 FROM nodes
 UNION ALL
 SELECT uid
 FROM ways) e;

 Answer:
 1600 unique users
```
##### 3. NUMBER OF NODES AND WAYS
```
SELECT COUNT(*) FROM nodes;
Answer: 3031834 nodes

SELECT COUNT(*) FROM ways;
Answer: 366755 ways
```
##### 4. NUMBER OF CHOSEN TYPE OF NODES
Number of poles in Houston:
```
SELECT COUNT(*)
FROM nodes_tags
WHERE value='pole';
Answer: 11190
```
Number of places of worship in Houston:
```
SELECT COUNT(*)
FROM nodes_tags
WHERE value='place_of_worship';
Answer: 2219
```

##### 5. ADDITIONAL STATISTICS
The Top 3 Users who made a 'pole' node:
```
SELECT user, COUNT(*) as num
FROM nodes, nodes_tags
WHERE nodes.id=nodes_tags.id and nodes_tags.value = 'pole'
GROUP BY user
ORDER BY num DESC
LIMIT 3;

Answer:
42429, 10564
Rallysta74, 379
beweta, 64
```

The Top 10 contributing users:
```
SELECT e.user, COUNT(*) as num
FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e
GROUP BY e.user
ORDER BY num DESC
LIMIT 10;

Answer:
woodpeck_fixbot,569278
TexasNHD,544483
afdreher,473340
scottyc,205303
cammace,193159
claysmalley,136594
brianboru,118555
skquinn,86265
RoadGeek_MD99,82255
Memoire,56679
```

##### 6. SUGGESTIONS ON IMPROVING DATA OR ANALYSIS
This dataset can be improved by partnering up with a game such as Pokemon Go to get
all users of the game involved in fixing the dataset. If there were certain spaces in the
map that were still unknown, Pokemon Go could just put some kind
of rare Pokemon or some kind of rare item incentive for the players to go to
that location and input the data about that place in order to get the incentive. This is a way to get all users of the game to help fix the map.

I feel like the input should also have restricted inputs as well.
If there were more drop down menus with 'street type' or had a drop down menu with a 5
number restricted zip code input then it would also automatically make the data cleaner.

### LIST OF REFERENCES:
I used Udacity course videos and Udacity Forums and www.sqlite.org to complete this
project.
