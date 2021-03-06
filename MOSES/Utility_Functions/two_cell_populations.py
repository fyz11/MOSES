#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:05:00 2018

@author: felix

Scripts to handle the analysis of two cell types used to primarily parse the different file names.

"""

import numpy as np 

def fetch_uniq_combos(stringlist):
    """ Script to return the enumeration of all possible pairwise string combinations of the input list of strings

    Parameters
    ----------
    stringlist : list or array of str
        list or array of strings to take the combination over

    Returns
    -------
    combos : numpy array
        array of all unique pairwise string combinations  
    """
    import itertools
    
    combos = []
    for i,j in itertools.combinations_with_replacement(np.arange(len(stringlist)), 2):
        combos.append([stringlist[i], stringlist[j]])
    combos = np.array(combos)

    return combos
    
# =============================================================================
#   Some scripts to collate all the metadata regarding the experiments
# =============================================================================
def match_combos_and_info(exp_info, uniq_combos, col1, col2):
    """ script to assign unique combination strings to parsed out table of experimental conditions

    Parameters
    ----------
    exp_info : Pandas table
        a table such as that generated by :func:`parse_exp_conditions_normal`, :func:`parse_exp_conditions_titrate` with meta information of each individual cell used in the video and their colours
    uniq_combos : 
        the unique standard all combination between all string names of cell types as generated by :func:`fetch_uniq_combos` above
    col1 : int
        column index of the input exp_info table giving the named string identity of the first cell type  
    col2 : int
        column index of the input exp_info table giving the named string identity of the second cell type  

    Returns
    -------
    exp_info : Pandas table
        an augmented table with three additional columns; 'CellIndex1' and 'CellIndex2' giving the position either 1 or 2 of each file name relative to the canonical naming of the combination and 'RefIndex' which specifies which canonical naming in uniq_combos is present

    """
    cells1 = exp_info.iloc[:,col1].values
    cells2 = exp_info.iloc[:,col2].values

    index1 = []
    index2 = []
    index_ref = []

    for i in range(len(cells1)):
        c1 = cells1[i]; c2=cells2[i]
        
        for j in range(len(uniq_combos)):
            ref_combo = uniq_combos[j]
            rc1 = ref_combo[0]; rc2 = ref_combo[1]
            
            if c1==rc1 and c2==rc2:
                index1.append(1) ; index2.append(2)
            elif c1==rc2 and c2==rc1:
                index1.append(2) ; index2.append(1)
            else:
#                print c1, c2
#                index1.append(np.nan) ; index2.append(np.nan)
                continue # not this ref. 
                
            index_ref.append(j)
        
#    print len(index1), len(index2), len(index_ref), len(cells1)
    exp_info.loc[:,'CellIndex1'] = index1
    exp_info.loc[:,'CellIndex2'] = index2
    exp_info.loc[:,'RefIndex'] = index_ref

    return exp_info

def parse_exp_conditions_normal(explist, split='_'):
    """ Utility script to parse the metadata naming convention used for our normal titration videos (no EGF addition) in the original eLife paper

    Given an example filename such as 'c_EPC2(G)_CP-A(R)_KSFM+RPMI_5_Fast GFP.tif_R-G.tif', this function will split the components according to the separator string used and parse out the experiment information such as cells and colour used.

    Parameters
    ----------
    explist : list or array-like
        list of string file names of each video file (not the absolute filepath)
    split : str
        the string separator used to separate the meta information present. In the example this is an underscore, '_'.

    Returns
    -------
    data_table : Pandas Table
        an output table summarising the experiment information provided by the filename namely, the media used, name of individual cell types and the respective colour channel corresponding to each cell i.e. ['Condition', 'Cell1', 'Color1', 'Cell2', 'Color2']

    """
    import pandas as pd 
    # we need to parse out iteratively... in a biased manner. 
    info = []

    for exp in explist:
        meta = (exp.strip()).split(split) # strip out white spaces... and then split. 
        cell1 = []
        cell2 = []
        condition = []        

        # iterate over the split to extract the information... 
        for met in meta:
            if 'KSFM' in met or 'RPMI' in met:
                condition.append(met)
            if 'CP-A' in met or 'EPC2' in met or 'OE33' in met or 'OE21' in met or 'AGS' in met:
                if cell1 == []:
                    cell1 = met
                elif cell2 == []:
                    cell2 = met
        
        # solve the condition problem 
        if len(condition) == 1:
            condition = condition[0]
        elif len(condition) > 1: 
            condition = '+'.join(condition)
            
        # solve the coloring issue for each cell    
        color1 = cell1.split('(')[1].strip('()')
        color2 = cell2.split('(')[1].strip('()')
        
        cell1 = cell1.split('(')[0]
        cell2 = cell2.split('(')[0]
        
        info.append([condition, cell1, color1, cell2, color2])
        
    info = np.array(info)
    data_table = pd.DataFrame(info, index = np.arange(info.shape[0]), columns=['Condition', 'Cell1', 'Color1', 'Cell2', 'Color2'])
        
    return data_table
    
    
def parse_exp_conditions_titrate(explist, exproot, split='_'):
    """ Utility script to parse the metadata naming convention used for our titration videos (not no addition) in the original eLife paper

    Given an example filename such as 'Tritration_EGF_KSFM_0dot5EGF_EPC2(R)_CP-A(G)_1_Fast GFP.tif_R-G..tif', this function will split the components according to the separator string used and parse out the experiment information such as cells and colour used. Unlike :func:`parse_exp_conditions_titrate_num` this version can handle 'dot' string in the concentration e.g. '0dot5EGF'.

    Parameters
    ----------
    explist : list or array-like
        list of string file names of each video file (not the absolute filepath)
    split : str
        the string separator used to separate the meta information present. In the example this is an underscore, '_'.

    Returns
    -------
    data_table : Pandas Table
        an output table summarising the experiment information provided by the filename namely, the media, stimulant, concentration, concentration unit used in addition to cell name and the respective colour channel of each cell i.e. ['Media', 'Stimulant', 'Condition', 'Units', 'Cell1', 'Color1', 'Cell2', 'Color2']

    """
    import pandas as pd 
    # we need to parse out iteratively... in a biased manner. 
    info = []

    for exp in explist:

        attempt = len(exp.split(exproot))
        
        if attempt == 1:
            meta = (exp.strip()).split(split)
        else:
            meta = (exp.split(exproot)[1].strip()).split(split) # strip out white spaces... and then split. 
        
        cell1 = []
        cell2 = []
        media = []   
        stimulant = []     
        condition = []
        conc_units = []

        # iterate over the split to extract the information... 
        ind = 0 
        for met in meta:
            # look for the media... 
            if 'KSFM' in met or 'RPMI' in met:
                media.append(met)
                
            # look for the cell type... this works well. 
            if 'CP-A' in met or 'EPC2' in met or 'OE33' in met or 'OE21' in met or 'AGS' in met:
                if cell1 == []:
                    cell1 = met
                elif cell2 == []:
                    cell2 = met
                    
             # look for the stimulant.
             
             # look for the concentration. 
            cond_markers = ['pg', 'ng', 'EGF', 'TFF3', 'GAST', 'FBS']

            for cond in cond_markers:
                if cond in met:
                    spl = met.split(cond)
                    
                    if len(spl) > 1:
                        conc = spl[0]

                        if len(conc) > 0:
                            conc_unit = cond
                            if conc_unit == 'pg' :
                                conc_units.append(conc_unit)
                                stimulant.append(meta[ind-1])
                            elif conc_unit == 'ng':
                                conc_units.append(conc_unit)
                                stimulant.append(meta[ind-1])
                            else:
                                stimulant.append(conc_unit)
                                conc_units.append('ng') # assume... 
    
                            if 'dot' in conc:
                                conc = float('.'.join(conc.split('dot')))
                                condition.append(conc)
#                                conc_units.append(conc_unit)
                            else:
                                conc = int(conc)
                                condition.append(conc)
#                                conc_units.append(conc_unit)
            ind += 1

        condition = '+'.join([str(c) for c in condition])
        conc_units = '+'.join([str(c) for c in conc_units])
        stimulant = '+'.join([str(c) for c in stimulant])
        
        # solve the condition problem 
        if len(media) == 1:
            media = media[0]
        elif len(media) > 1: 
            media = '+'.join(media)
            
            
        if media == []:
            media = 'KSFM+RPMI'
            
        if stimulant == []:
            stimulant = 'EGF'
                      
        # solve the coloring issue for each cell    
        color1 = cell1.split('(')[1].strip('()')
        color2 = cell2.split('(')[1].strip('()')
        
        cell1 = cell1.split('(')[0]
        cell2 = cell2.split('(')[0]
        
        info.append([media, stimulant, condition, conc_units, cell1, color1, cell2, color2])
    info = np.array(info)
    data_table = pd.DataFrame(info, index = np.arange(info.shape[0]), columns=['Media', 'Stimulant', 'Condition', 'Units', 'Cell1', 'Color1', 'Cell2', 'Color2'])
        
    return data_table   
    
def parse_exp_conditions_titrate_num(explist, exproot, split='_'):
    """ Utility script to parse the metadata naming convention used for our titration videos (not no addition) in the original eLife paper

    Given an example filename such as 'Tritration_EGF_KSFM_5EGF_EPC2(R)_CP-A(G)_1_Fast GFP.tif_R-G..tif', this function will split the components according to the separator string used and parse out the experiment information such as cells and colour used. Unlike :func:`parse_exp_conditions_titrate` this version requires the concentration in front of the stimulate to be integer e.g. '5EGF'.

    Parameters
    ----------
    explist : list or array-like
        list of string file names of each video file (not the absolute filepath)
    split : str
        the string separator used to separate the meta information present. In the example this is an underscore, '_'.

    Returns
    -------
    data_table : Pandas Table
        an output table summarising the experiment information provided by the filename namely, the media, stimulant, concentration, concentration unit used in addition to cell name and the respective colour channel of each cell i.e. ['Media', 'Stimulant', 'Condition', 'Units', 'Cell1', 'Color1', 'Cell2', 'Color2']

    """
    import pandas as pd 
    # we need to parse out iteratively... in a biased manner. 
    info = []
    for exp in explist:

        attempt = len(exp.split(exproot))
        
        if attempt == 1:
            meta = (exp.strip()).split(split)
        else:
            meta = (exp.split(exproot)[1].strip()).split(split) # strip out white spaces... and then split. 
        
        cell1 = []
        cell2 = []
        media = []   
        stimulant = []     
        condition = []
        conc_units = []

        # iterate over the split to extract the information... 
        ind = 0 
        for met in meta:
            # look for the media... 
            if 'KSFM' in met or 'RPMI' in met:
                media.append(met)
                
            # look for the cell type... this works well. 
            if 'CP-A' in met or 'EPC2' in met or 'OE33' in met or 'OE21' in met or 'AGS' in met:
                if cell1 == []:
                    cell1 = met
                elif cell2 == []:
                    cell2 = met
                    
             # look for the concentration to find the stimulant and units. 
            if met.isdigit():
                condition.append(met)
                stimulant.append(meta[ind-1])
                conc_units.append('ng')
            ind += 1

        condition = '+'.join([str(c) for c in condition])
        conc_units = '+'.join([str(c) for c in conc_units])
        stimulant = '+'.join([str(c) for c in stimulant])
        
        # solve the condition problem 
        if len(media) == 1:
            media = media[0]
        elif len(media) > 1: 
            media = '+'.join(media)
            
        if media == []:
            media = 'KSFM+RPMI'
            
        if stimulant == []:
            stimulant = 'EGF'
                      
        # solve the coloring issue for each cell    
        color1 = cell1.split('(')[1].strip('()')
        color2 = cell2.split('(')[1].strip('()')
        
        cell1 = cell1.split('(')[0]
        cell2 = cell2.split('(')[0]
        
        info.append([media, stimulant, condition, conc_units, cell1, color1, cell2, color2])
    info = np.array(info)
    data_table = pd.DataFrame(info, index = np.arange(info.shape[0]), columns=['Media', 'Stimulant', 'Condition', 'Units', 'Cell1', 'Color1', 'Cell2', 'Color2'])
        
    return data_table   



    