l1 = [
'PERS', 'EVENT', 'CARDINAL',
'NORP', 'DATE', 'ORDINAL',
'OCC', 'TIME', 'PERCENT',
'ORG', 'LANGUAGE', 'QUANTITY',
'GPE', 'WEBSITE', 'UNIT',
'LOC', 'LAW', 'MONEY',
'FAC', 'PRODUCT', 'CURR',
]

l2 = [
    #GPE
    "COUNTRY", "STATE-OR-PROVINCE", "TOWN",
    "NEIGHBORHOOD","CAMP","GPE_ORG", "SPORT",
    #LOC
    "CONTINENT", "CLUSTER","ADDRESS","BOUNDARY",
    "CELESTIAL", "WATER-BODY", "LAND-REGION-NATURAL",
    "REGION-GENERAL", "REGION-INTERNATIONAL",
    #ORG
    "GOV", "COM", "EDU", "ENT",
    "NONGOV", "MED", "REL", "SCI",
    "SPO","ORG_FAC",
    #FAC
    "PLANT", "AIRPORT", "BUILDING-OR-GROUNDS",
    "SUBAREA-FACILITY", "PATH",
]

l3 = [
    #GPE_ORG
    "COUNTRY", "STATE-OR-PROVINCE", "TOWN",
    "NEIGHBORHOOD","CAMP", "SPORT",
    "GOV", "COM", "EDU", "ENT",
    "NONGOV", "MED", "REL", "SCI",
    "SPO","ORG_FAC",
    #ORG_FAC
    "GOV", "COM", "EDU", "ENT",
    "NONGOV", "MED", "REL", "SCI",
    "SPO",
    "PLANT", "AIRPORT", "BUILDING-OR-GROUNDS",
    "SUBAREA-FACILITY", "PATH"
]

def get_IOB_labels():
    l1_IOB = []
    l1_IOB.append('O')
    for i in l1:
        l1_IOB.append('B-'+i)
        l1_IOB.append('I-'+i)


    l2_IOB = []
    l2_IOB.append('O')
    for i in l2:
        l2_IOB.append('B-'+i)
        l2_IOB.append('I-'+i)

    main_label_map = {label: i for i, label in enumerate(l1_IOB)}
    subtype_label_map = {label: i for i, label in enumerate(l2_IOB)}

    inv_main_label_map = {i: label for i, label in enumerate(l1_IOB)}
    inv_subtype_label_map = {i: label for i, label in enumerate(l2_IOB)}

    return main_label_map, subtype_label_map, inv_main_label_map, inv_subtype_label_map