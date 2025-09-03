# In MIMIC-III, diagnosis codes are ICD-9
# •	Format:
# •	Numeric codes (001–999)
# •	Alphanumeric codes (E and V codes)
# •	3-digit category code, with optional 1 or 2 decimal digits for sub-classification
# •	NNN.xx or VNN.xx or ENN.xx

def expand_level2():
    # map each ICD-9 code to a range
    # return 
    # {
        #   '001': '001-009',
        #   '002': '001-009',
        #   ...
        #   '250': '250-259',
        #   ...
        #   'V10': 'V10-V19',
        #   ...
        #   'E880': 'E880-E888',
    # }
    
    level2 = ['001-009', '010-018', '020-027', '030-041', '042', '045-049', '050-059', '060-066', '070-079', '080-088',
              '090-099', '100-104', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165',
              '170-176',
              '176', '179-189', '190-199', '200-208', '209', '210-229', '230-234', '235-238', '239', '240-246',
              '249-259',
              '260-269', '270-279', '280-289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338',
              '339', '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417',
              '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508',
              '510-519',
              '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599',
              '600-608',
              '610-611', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679',
              '680-686',
              '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740-759', '760-763', '764-779',
              '780-789',
              '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854',
              '860-869',
              '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939',
              '940-949',
              '950-957', '958-959', '960-979', '980-989', '990-995', '996-999', 'V01-V91', 'V01-V09', 'V10-V19',
              'V20-V29',
              'V30-V39', 'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82', 'V83-V84', 'V85', 'V86', 'V87', 'V88', 'V89',
              'V90',
              'V91', 'E000-E899', 'E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825', 'E826-E829', 'E830-E838',
              'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879', 'E880-E888', 'E890-E899',
              'E900-E909', 'E910-E915', 'E916-E928', 'E929', 'E930-E949', 'E950-E959', 'E960-E969', 'E970-E978',
              'E980-E989', 'E990-E999']

    level2_expand = {}
    for i in level2:
        tokens = i.split('-')
        if i[0] == 'V':
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["V%02d" % j] = i
        elif i[0] == 'E':
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["E%03d" % j] = i
        else:
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0]), int(tokens[1]) + 1):
                    level2_expand["%03d" % j] = i
    return level2_expand


# for diagnosis
def build_icd9_tree(diag_tree_voc, unique_codes):
    code2sentence = {}

    level3_dict = expand_level2()
    for code in unique_codes:
        level1 = code[5:] # code itself, remove DIA_
        level2 = level1[:4] if level1[0] == 'E' else level1[:3] # first 3 or 4 characters of the code
        level3 = level3_dict[level2] # the range of codes that level2 belongs to

        sample = [level1, level2, level3]

        diag_tree_voc.add_sentence(sample)
        code2sentence[code] = sample

    # code2sentence = {code: [level1, level2, level3, level4] for code in unique_codes}
    # icd9_tree_voc is a Voc object that contains all unique level1, level2, level3, level4 codes (all in one voc)
    return code2sentence, diag_tree_voc

# for medication
def build_atc_tree(graph_voc, unique_codes):
    code2sentence = {}
    for code in unique_codes:
        # sample = [code] + [code[:i] for i in [4, 3, 1]] + [root_node]
        sample = [code[:i] for i in [4, 3, 1]]
        graph_voc.add_sentence(sample)
        code2sentence[code] = sample
    
    return code2sentence, graph_voc