# no_internet_rate is "Households with no internet use in and outside the home in 2010", gathered from http://www.ntia.doc.gov/files/ntia/data/CPS2010Tables/t11_2.txt
STATES = {"AL":{"full_name":"ALABAMA", 
			"geo_region":"South",
			"no_internet_rate":25.82},
		"AK":{"full_name":"ALASKA", 
			"geo_region":"West",
			"no_internet_rate":11.36},
		"AS":{"full_name":"AMERICAN SAMOA", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"AZ":{"full_name":"ARIZONA", 
			"geo_region":"West",
			"no_internet_rate":16.54},
		"AR":{"full_name":"ARKANSAS", 
			"geo_region":"South",
			"no_internet_rate":29.13},
		"CA":{"full_name":"CALIFORNIA", 
			"geo_region":"West",
			"no_internet_rate":15.81},
		"CO":{"full_name":"COLORADO", 
			"geo_region":"West",
			"no_internet_rate":17.32},
		"CT":{"full_name":"CONNECTICUT", 
			"geo_region":"Northeast",
			"no_internet_rate":18.05},
		"DE":{"full_name":"DELAWARE", 
			"geo_region":"South",
			"no_internet_rate":20.92},
		"DC":{"full_name":"DISTRICT OF COLUMBIA", 
			"geo_region":"Northeast",
			"no_internet_rate":19.05},
		"FM":{"full_name":"FEDERATED STATES OF MICRONESIA", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"FL":{"full_name":"FLORIDA", 
			"geo_region":"South",
			"no_internet_rate":20.07},
		"GA":{"full_name":"GEORGIA", 
			"geo_region":"South",
			"no_internet_rate":20.11},
		"GU":{"full_name":"GUAM GU", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"HI":{"full_name":"HAWAII", 
			"geo_region":"Outer",
			"no_internet_rate":21.43},
		"ID":{"full_name":"IDAHO", 
			"geo_region":"West",
			"no_internet_rate":15.88},
		"IL":{"full_name":"ILLINOIS", 
			"geo_region":"Midwest",
			"no_internet_rate":20.15},
		"IN":{"full_name":"INDIANA", 
			"geo_region":"Midwest",
			"no_internet_rate":25.27},
		"IA":{"full_name":"IOWA", 
			"geo_region":"Midwest",
			"no_internet_rate":20.55},
		"KS":{"full_name":"KANSAS", 
			"geo_region":"Midwest",
			"no_internet_rate":15.22},
		"KY":{"full_name":"KENTUCKY", 
			"geo_region":"South",
			"no_internet_rate":27.98},
		"LA":{"full_name":"LOUISIANA", 
			"geo_region":"South",
			"no_internet_rate":25.06},
		"ME":{"full_name":"MAINE", 
			"geo_region":"Northeast",
			"no_internet_rate":18.28},
		"MH":{"full_name":"MARSHALL ISLANDS", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"MD":{"full_name":"MARYLAND", 
			"geo_region":"South",
			"no_internet_rate":16.75},
		"MA":{"full_name":"MASSACHUSETTS", 
			"geo_region":"Northeast",
			"no_internet_rate":16.18},
		"MI":{"full_name":"MICHIGAN", 
			"geo_region":"Midwest",
			"no_internet_rate":19.19},
		"MN":{"full_name":"MINNESOTA", 
			"geo_region":"Midwest",
			"no_internet_rate":16.56},
		"MS":{"full_name":"MISSISSIPPI", 
			"geo_region":"South",
			"no_internet_rate":28.57},
		"MO":{"full_name":"MISSOURI", 
			"geo_region":"Midwest",
			"no_internet_rate":21.79},
		"MT":{"full_name":"MONTANA", 
			"geo_region":"West",
			"no_internet_rate":24.26},
		"NE":{"full_name":"NEBRASKA", 
			"geo_region":"Midwest",
			"no_internet_rate":17.46},
		"NV":{"full_name":"NEVADA", 
			"geo_region":"West",
			"no_internet_rate":15.67},
		"NH":{"full_name":"NEW HAMPSHIRE", 
			"geo_region":"Northeast",
			"no_internet_rate":13.65},
		"NJ":{"full_name":"NEW JERSEY", 
			"geo_region":"Northeast",
			"no_internet_rate":17.14},
		"NM":{"full_name":"NEW MEXICO", 
			"geo_region":"West",
			"no_internet_rate":23.23},
		"NY":{"full_name":"NEW YORK", 
			"geo_region":"Northeast",
			"no_internet_rate":20.70},
		"NC":{"full_name":"NORTH CAROLINA", 
			"geo_region":"South",
			"no_internet_rate":23.47},
		"ND":{"full_name":"NORTH DAKOTA", 
			"geo_region":"Midwest",
			"no_internet_rate":20.13},
		"MP":{"full_name":"NORTHERN MARIANA ISLANDS", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"OH":{"full_name":"OHIO", 
			"geo_region":"Midwest",
			"no_internet_rate":21.56},
		"OK":{"full_name":"OKLAHOMA", 
			"geo_region":"South",
			"no_internet_rate":22.70},
		"OR":{"full_name":"OREGON", 
			"geo_region":"West",
			"no_internet_rate":13.82},
		"PW":{"full_name":"PALAU", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"PA":{"full_name":"PENNSYLVANIA", 
			"geo_region":"Northeast",
			"no_internet_rate":21.87},
		"PR":{"full_name":"PUERTO RICO", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"RI":{"full_name":"RHODE ISLAND", 
			"geo_region":"Northeast",
			"no_internet_rate":20.16},
		"SC":{"full_name":"SOUTH CAROLINA", 
			"geo_region":"South",
			"no_internet_rate":25.62},
		"SD":{"full_name":"SOUTH DAKOTA", 
			"geo_region":"Midwest",
			"no_internet_rate":19.03},
		"TN":{"full_name":"TENNESSEE", 
			"geo_region":"South",
			"no_internet_rate":27.80},
		"TX":{"full_name":"TEXAS", 
			"geo_region":"South",
			"no_internet_rate":19.77},
		"UT":{"full_name":"UTAH", 
			"geo_region":"West",
			"no_internet_rate":9.90},
		"VT":{"full_name":"VERMONT", 
			"geo_region":"Northeast",
			"no_internet_rate":16.48},
		"VI":{"full_name":"VIRGIN ISLANDS", 
			"geo_region":"Outer",
			"no_internet_rate":None},
		"VA":{"full_name":"VIRGINIA", 
			"geo_region":"South",
			"no_internet_rate":20.16},
		"WA":{"full_name":"WASHINGTON", 
			"geo_region":"West",
			"no_internet_rate":11.63},
		"WV":{"full_name":"WEST VIRGINIA", 
			"geo_region":"South",
			"no_internet_rate":27.13},
		"WI":{"full_name":"WISCONSIN", 
			"geo_region":"Midwest",
			"no_internet_rate":16.85},
		"WY":{"full_name":"WYOMING", 
			"geo_region":"West",
			"no_internet_rate":15.65}
}
REGION = {abbr:STATES[abbr]["geo_region"]for abbr in STATES}
NONETRATE = {abbr:STATES[abbr]["no_internet_rate"]for abbr in STATES}