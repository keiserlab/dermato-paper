# I/O Handling
from __future__ import print_function
import csv,glob,re
import os,sys

# Data handling
import numpy as np

# Standard Tools
from collections import OrderedDict
from itertools import zip_longest

# Custom functions
##from image_utils import *
##from segment_and_extract import *
##from align_functions import *

# Widgets + Plotting
import traitlets
import ipywidgets as widgets
from ipywidgets import Text, Label
from ipywidgets import Layout, HBox, VBox, Box
from ipywidgets import Button, Accordion
import matplotlib.pyplot as plt


'''
#############################################################################################################
#                                          Observe/Update Widgets                                              
#############################################################################################################
'''

def observe_widget(wdgt, pname, ptype, fx):
    """Helper function for subclassing a widget with pname and ptype,
    and assigning it to observe function fx"""
    wdgt.pname = pname
    wdgt.ptype = ptype
    wdgt.observe(fx)
    return


def update_thresholds(new_thresh_val, description, min_slider, max_slider):
    """Helper function for managing which slider (min-thresh vs max-thresh) to update"""
    minThresh = min_slider.value
    maxThresh = max_slider.value
    if description == 'Min-Thresh':
        minThresh = new_thresh_val
    elif description == 'Max-Thresh':
        maxThresh = new_thresh_val
    return minThresh, maxThresh




'''
#############################################################################################################
#                                          Checkbox/Label Widgets                                              
#############################################################################################################
'''

def get_HE_IHC_segment_labels(he_moniker, seg_regions_HE, ihc_moniker, seg_regions_IHC):
    """Generate set of HE and IHC segment labels, and return sorted lists of each."""
    HE_labels = set()
    IHC_labels = set()
    he_shortname = he_moniker.split('-')[-1]
    ihc_shortname = ihc_moniker.split('-')[-1]
    paired_regions = list(zip_longest(seg_regions_HE, seg_regions_IHC, fillvalue=None))
    for i,pair in enumerate(paired_regions):
        if pair[0] is None:
            IHC_seg_label = '{}_seg.{}'.format(ihc_shortname, i+1)
            IHC_labels.add(IHC_seg_label)
        elif pair[1] is None:
            HE_seg_label = '{}_seg.{}'.format(he_shortname, i+1)
            HE_labels.add(HE_seg_label)
        else:
            HE_seg_label = '{}_seg.{}'.format(he_shortname, i+1)
            IHC_seg_label = '{}_seg.{}'.format(ihc_shortname, i+1)
            HE_labels.add(HE_seg_label)
            IHC_labels.add(IHC_seg_label)
    sorted_HE = sorted(HE_labels, key= lambda x:(int(x.split('.')[-1])) )
    sorted_IHC = sorted(IHC_labels, key= lambda x:(int(x.split('.')[-1])) )
    return sorted_HE, sorted_IHC


def collect_HE_wparams(wdict):
    """Collects widget parameters for HE image"""
    HE = wdict['HE']
    HE_chroma = wdict['HE_chroma'].value
    HE_minThresh = wdict['HE_minslider'].value
    HE_maxThresh = wdict['HE_maxslider'].value
    return HE, HE_chroma, HE_minThresh, HE_maxThresh


def collect_IHC_wparams(wdict):
    """Collects widget parameters for IHC image"""
    IHC = wdict['IHC']
    IHC_chroma = wdict['IHC_chroma'].value
    IHC_minThresh = wdict['IHC_minslider'].value
    IHC_maxThresh = wdict['IHC_maxslider'].value
    IHC_rotation = wdict['IHC_rotation'].value
    return IHC, IHC_chroma, IHC_minThresh, IHC_maxThresh, IHC_rotation


def collect_matcher_wparams(wdict):
    """Collects labels and corresponding childen from segment matcher tab widget"""
    matcher_items = wdict['segment_matcher']._titles.items()
    sorted_items = sorted(matcher_items, key=lambda k: int(k[0]))
    match_labels = [label for idx,label in sorted_items]
    match_children = wdict['segment_matcher'].children
    return match_labels, match_children



'''
#############################################################################################################
#                                              Tab Widgets                                           
#############################################################################################################
'''

def construct_segment_matcher_objects(HE_labels, IHC_labels):
    """Constructs widget contents (tab-labels) and children for segment-matcher tab widget."""
    # Construct tab contents
    align_tab = ['Segment']
    tab_contents = HE_labels + align_tab
    # Construct tab chidlren (IHC radio-button options and alignment button)
    opts = IHC_labels + ['None']
    match_dict = dict(zip_longest(HE_labels, IHC_labels, fillvalue='None'))
    radio_buttons = [widgets.RadioButtons(options=opts, value=match_dict[label],
                                          description='IHC Pair', disabled=False) for label in HE_labels]
    align_button = [widgets.ToggleButton(description='Segment Pairs', disabled=False, 
                                         button_style='', tooltip='Segment and extract sections')]
    children = radio_buttons + align_button
    return tab_contents, children


def create_tab_widget(tab_contents, children):
    tab = widgets.Tab()
    tab.children = children
    for i,label in enumerate(tab_contents):
        tab.set_title(i, label)
    return tab


#def update_tab_widget(tab, tab_contents, children):
#    tab.children = children
#    for i,label in enumerate(tab_contents):
#        tab.set_title(i, label)
#    return tab


def update_tab_widget(tab, new_content, children):
    content_range = set([str(i) for i,_ in enumerate(new_content)])
    tab_keys = set(tab._titles.keys())
    diff = tab_keys - content_range
    if not diff:
        return
    #tab.children = children
    for ex in diff:
        tab._titles.pop(ex)
    #for i,label in enumerate(new_content):
    #    tab.set_title(i, label)

