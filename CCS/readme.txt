https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp

We are using the icd9-to-ccs mapping to group the icd9 codes into a smaller set.
This is necessary because the number of codes defines the cardinality of one of the tensor dimensions, which easy explodes the memory.

About the files in this directory: https://www.hcup-us.ahrq.gov/toolssoftware/ccs/CCSUsersGuide.pdf


CCS$DXREF 2015.csvTranslation file that maps ICD-9-CM diagnosis codes into single-level CCS diagnosis categories, with complete information about each ICD-9-CM code and brief CCS labels.
DXLABEL 2015.csvLabel file contains the complete descriptive single-level CCS diagnosis category names to use when reporting the diagnosis categories. Category 158 label was changed in 2013.

