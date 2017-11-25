#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

osm_file = open("houston_texas.osm", "r")

street_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
street_types = defaultdict(set)
post_codes_set = set()

expected = ['Street', 'Avenue', 'Boulevard', 'Drive', 'Court', 'Place', 'Square',
			'Lane', 'Road', 'Trail', 'Parkway', 'Commons', 'Circle', 'Freeway',
			'Highway', 'Plaza', 'Way']

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
        	street_types[street_type].add(street_name)



def print_sorted_dict(d):
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print('{}: {}'.format(k, v))

def fix_postcode(code):
	if len(code) >= 5:
		start_postcode = code.find('7')
		end_postcode = start_postcode + 5
		new_code = code[start_postcode:end_postcode]
		return new_code


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit():
	for event, elem in ET.iterparse(osm_file, events=("start",)):
		if elem.tag == 'way' or elem.tag == 'node':
			for tag in elem.iter('tag'):
				if tag.attrib['k'] == 'addr:postcode':
					new_codes = fix_postcode(tag.attrib['v'])
					post_codes_set.add(new_codes)
					print(new_codes) 

if __name__ == '__main__':
    audit()
