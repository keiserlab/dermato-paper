# I/O Handling
import os,sys
import glob,re

# Data handling
import numpy as np

# Standard Tools
from collections import OrderedDict

# Custom functions
from image_utils import natural_sort_key, grouper
from widget_utils import observe_widget, update_thresholds
from widget_utils import get_HE_IHC_segment_labels, construct_segment_matcher_objects
from widget_utils import create_tab_widget, update_tab_widget
from segment_and_extract import gen_mask_output


# Widgets + Plotting
import traitlets
import ipywidgets as widgets
from ipywidgets import Text, Label
from ipywidgets import Layout, HBox, VBox, Box
from ipywidgets import Button, Accordion
import matplotlib.pyplot as plt


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


def preview_segmentations(imDir, imFn_pattern, wdgts=None, heLevel=2, ihcLevel=2, default_min=0.13, default_max=1.0, 
                          default_rotation=None):
    """Displays preview of WSI segmentations with widgets for adjusting seg-parameter thresholds and
    alignments, for each of the WSI files matched by imDir and imFn_pattern. Updates wdgts along the way."""

    HE_OPTIONS = ['HE', 'HEL', 'HE2', 'HEL2']
    IHC_OPTIONS = ['MELA', 'MelPro', 'MITF', 'MITFR', 'SOX10', 'SOX10R', 'p16', 'HMB485']

    # Initialize wdgts if needed
    if wdgts is None:
        wdgts = {}

    # Initialize layouts for each preview sub-window
    pair_layout = Layout(display='flex', flex_flow='row', justify_content='flex-start', width='15em')
    params_layout = Layout(display='flex', flex_flow='row', justify_content='space-around', width='95%')
    hist_layout = Layout(display='flex', flex_flow='row', justify_content='center', width='95%', align_items='stretch')
    tab_layout = Layout(display='flex', flex_flow='row', justify_content='flex-start', width='95%')
    display_layout = Layout(display='flex', flex_flow='column', align_items='stretch')

    def _on_chroma_change(floatText):
        """Updates preview to reflect change in chroma option"""
        if floatText['type'] == 'change' and floatText['name'] == 'value':
            chroma = floatText.new
            pname = floatText.owner.pname
            ptype = floatText.owner.ptype
            if ptype in HE_OPTIONS:
                minThresh = wdgts[pname]['HE_minslider'].value
                maxThresh = wdgts[pname]['HE_maxslider'].value
                level = wdgts[pname]['HE_level'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation=None, level=level)
            elif ptype in IHC_OPTIONS:
                minThresh = wdgts[pname]['IHC_minslider'].value
                maxThresh = wdgts[pname]['IHC_maxslider'].value
                level = wdgts[pname]['IHC_level'].value
                rotation = wdgts[pname]['IHC_rotation'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation, level=level)
        return

    def _on_threshold_change(sldr):
        """Updates preview to reflect change in slider option"""
        if sldr['type'] == 'change' and sldr['name'] == 'value':
            val = sldr.new
            desc = sldr.owner.description
            pname = sldr.owner.pname
            ptype = sldr.owner.ptype
            if ptype in HE_OPTIONS:
                chroma = wdgts[pname]['HE_chroma'].value
                minThresh,maxThresh = update_thresholds(val, desc, wdgts[pname]['HE_minslider'], 
                                                        wdgts[pname]['HE_maxslider'])
                level = wdgts[pname]['HE_level'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation=None, level=level)
            elif ptype in IHC_OPTIONS:
                chroma = wdgts[pname]['IHC_chroma'].value
                minThresh,maxThresh = update_thresholds(val, desc, wdgts[pname]['IHC_minslider'], 
                                                        wdgts[pname]['IHC_maxslider'])
                level = wdgts[pname]['IHC_level'].value
                rotation = wdgts[pname]['IHC_rotation'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation, level=level)
        return

    def _on_level_change(intText):
        """Updates preview to reflect change in slider option"""
        if intText['type'] == 'change' and intText['name'] == 'value':
            level = intText.new
            pname = intText.owner.pname
            ptype = intText.owner.ptype
            if ptype in HE_OPTIONS:
                chroma = wdgts[pname]['HE_chroma'].value
                minThresh = wdgts[pname]['HE_minslider'].value
                maxThresh = wdgts[pname]['HE_maxslider'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation=None, level=level)
            if ptype in IHC_OPTIONS:
                chroma = wdgts[pname]['IHC_chroma'].value
                minThresh = wdgts[pname]['IHC_minslider'].value
                maxThresh = wdgts[pname]['IHC_maxslider'].value
                rotation = wdgts[pname]['IHC_rotation'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation, level=level)
        return

    def _on_rotation_change(dropDown):
        """Update rotation of IHC slide if dropdown changed"""
        if dropDown['type'] == 'change' and dropDown['name'] == 'value':
            rotation = dropDown.new
            pname = dropDown.owner.pname
            ptype = dropDown.owner.ptype
            if ptype in HE_OPTIONS:
                chroma = wdgts[pname]['HE_chroma'].value
                minThresh = wdgts[pname]['HE_minslider'].value
                maxThresh = wdgts[pname]['HE_maxslider'].value
                level = wdgts[pname]['HE_level'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation=None, level=level)
            if ptype in IHC_OPTIONS:
                chroma = wdgts[pname]['IHC_chroma'].value
                minThresh = wdgts[pname]['IHC_minslider'].value
                maxThresh = wdgts[pname]['IHC_maxslider'].value
                level = wdgts[pname]['IHC_level'].value
                _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation, level=level)
        return

    def _update_output(pname, ptype, chroma, minThresh, maxThresh, rotation=None, level=2):
        """Update HE and IHC visual output based on parameter updates"""
        preview = wdgts[pname]
        heFN = preview['HE']
        ihcFN = preview['IHC']
        # If update happened to HE params, update HE mask objects with new values
        # Update IHC with stored values to prevent labeling errors in segment matcher
        if ptype in HE_OPTIONS:
            preview['HE_histout'].clear_output()
            with preview['HE_histout']:
                HE_mask_objects = gen_mask_output(heFN, chroma, minThresh, maxThresh, rotation=None, level=level)
                HE_regions, HE_bwmasks, HE_comasks, imW, imH = HE_mask_objects
                IHC_mask_objects = gen_mask_output(ihcFN, preview['IHC_chroma'].value, preview['IHC_minslider'].value, 
                                                          preview['IHC_maxslider'].value, preview['IHC_rotation'].value, 
                                                          preview['IHC_level'].value, verbose=False)
                IHC_regions, IHC_bwmasks, IHC_comasks, imW, imH = IHC_mask_objects
        # If update happened to IHC params, update IHC mask objects with new values
        # Update HE with stored values to prevent labeling errors in segment matcher
        elif ptype in IHC_OPTIONS:
            preview['IHC_histout'].clear_output()
            with preview['IHC_histout']:
                HE_mask_objects = gen_mask_output(heFN, preview['HE_chroma'].value, preview['HE_minslider'].value, 
                                                        preview['HE_maxslider'].value, rotation=None, 
                                                        level=preview['HE_level'].value, verbose=False)
                HE_regions, HE_bwmasks, HE_comasks, imW, imH = HE_mask_objects
                IHC_mask_objects = gen_mask_output(ihcFN, chroma, minThresh, maxThresh, rotation, level=level)
                IHC_regions, IHC_bwmasks, IHC_comasks, imW, imH = IHC_mask_objects
        else:
            print('pair-type: {} not able to update'.format(ptype.split('-')[-1]))
            return
        # Update segment matches to reflect preview changes
        he_moniker = pname.split('+')[0]
        ihc_moniker = pname.split('+')[1]
        HE_labels, IHC_labels = get_HE_IHC_segment_labels(he_moniker, HE_regions, ihc_moniker, IHC_regions)
        tab_content, new_children = construct_segment_matcher_objects(HE_labels, IHC_labels)
        update_tab_widget(wdgts[pname]['segment_matcher'], tab_content, children)
        #new_matcher = create_tab_widget(tab_contents, children)
        wdgts[pname]['segment_matcher'].children = new_children
        for i,label in enumerate(tab_content):
            wdgts[pname]['segment_matcher'].set_title(i, label)
        return
    


    

    # Match WSI file paths from pattern, loop through files, create preview items from dictionary
    path_pattern = os.path.join(imDir, imFn_pattern)
    wsi_files = sorted(glob.glob(path_pattern), key=natural_sort_key)
    wsi_pairs = grouper(wsi_files, 2)
    for heFN,ihcFN in wsi_pairs:
        he_moniker = os.path.basename(heFN).split('.')[0]
        ihc_moniker = os.path.basename(ihcFN).split('.')[0]
        pair_label = '+'.join([he_moniker, ihc_moniker])
        wdgts[pair_label] = dict(pair_chbox=widgets.Checkbox(value=True, description=pair_label, disabled=False, indent=False, 
                                                             layout=Layout(flex='1 1 auto', width='15em')),
                                 HE=heFN,
                                 HE_chroma=widgets.BoundedFloatText(value=4., min=0., max=14., description='Chroma:', disabled=False, 
                                                                    layout=Layout(width='9em')),
                                 HE_minslider=widgets.FloatSlider(description='Min-Thresh', value=default_min, min=0., max=1., step=0.01, 
                                                                  readout=True, orientation='horizontal', continuous_update=False, 
                                                                  layout=Layout(flex='2 1 auto', width='auto')),
                                 HE_maxslider=widgets.FloatSlider(description='Max-Thresh', value=default_max, min=.50, max=1.0, step=0.01, 
                                                                  readout=True, orientation='horizontal', continuous_update=False, 
                                                                  layout=Layout(flex='1 1 auto', width='auto')),
                                 HE_level=widgets.IntText(value=heLevel, description='WSI level:', disabled=False, 
                                                          layout=Layout(width='8em')),
                                 HE_histout=widgets.Output(),
                                 IHC=ihcFN,
                                 IHC_chroma=widgets.BoundedFloatText(value=1., min=0., max=14., description='Chroma:', disabled=False,
                                                                     layout=Layout(width='9em')),
                                 IHC_minslider=widgets.FloatSlider(description='Min-Thresh', value=default_min, min=0., max=1., step=0.01, 
                                                                   readout=True, orientation='horizontal', continuous_update=False, 
                                                                   layout=Layout(flex='2 1 auto', width='auto')),
                                 IHC_maxslider=widgets.FloatSlider(description='Max-Thresh', value=default_max, min=.50, max=1.0, step=0.01, 
                                                                   readout=True, orientation='horizontal', continuous_update=False, 
                                                                   layout=Layout(flex='1 1 auto', width='auto')),
                                 IHC_level=widgets.IntText(value=ihcLevel, description='WSI level:', disabled=False, 
                                                           layout=Layout(width='8em')),
                                 IHC_rotation=widgets.Dropdown(options=OrderedDict([('None', None), ('90', 90), ('180', 180), ('270', 270)]), 
                                                             value=None, description='rotation', disabled=False,
                                                             layout=Layout(flex='1 1 auto', width='auto')),
                                 IHC_histout=widgets.Output(),
                                 segment_matcher=None,
                                 display_box=None)

        
        # Create default HE segmentation output
        HE_chroma = wdgts[pair_label]['HE_chroma'].value
        with wdgts[pair_label]['HE_histout']:
            HE_regions, HE_bwmasks, HE_comasks, imW, imH = gen_mask_output(heFN, HE_chroma, default_min, default_max, 
                                                                           default_rotation, heLevel)

        # Create default IHC segmentation output
        IHC_chroma = wdgts[pair_label]['IHC_chroma'].value
        with wdgts[pair_label]['IHC_histout']:
            IHC_regions, IHC_bwmasks, IHC_comasks, imW, imH = gen_mask_output(ihcFN, IHC_chroma, default_min, default_max, 
                                                                              default_rotation, ihcLevel)

        # Generate default segment-matcher tab
        HE_labels, IHC_labels = get_HE_IHC_segment_labels(he_moniker, HE_regions, ihc_moniker, IHC_regions)
        tab_contents, children = construct_segment_matcher_objects(HE_labels, IHC_labels)
        seg_matcher = create_tab_widget(tab_contents, children)
        wdgts[pair_label]['segment_matcher'] = seg_matcher


        # Update HE params on change
        HE_shortname = he_moniker.split('-')[-1]
        IHC_shortname = ihc_moniker.split('-')[-1]
        observe_widget(wdgts[pair_label]['HE_chroma'], pair_label, HE_shortname, _on_chroma_change)
        observe_widget(wdgts[pair_label]['HE_minslider'], pair_label, HE_shortname, _on_threshold_change)
        observe_widget(wdgts[pair_label]['HE_maxslider'], pair_label, HE_shortname, _on_threshold_change)
        observe_widget(wdgts[pair_label]['HE_level'], pair_label, HE_shortname, _on_level_change)
        # Update IHC params on change
        observe_widget(wdgts[pair_label]['IHC_chroma'], pair_label, IHC_shortname, _on_chroma_change)
        observe_widget(wdgts[pair_label]['IHC_minslider'], pair_label, IHC_shortname, _on_threshold_change)
        observe_widget(wdgts[pair_label]['IHC_maxslider'], pair_label, IHC_shortname, _on_threshold_change)
        observe_widget(wdgts[pair_label]['IHC_level'], pair_label, IHC_shortname, _on_level_change)
        observe_widget(wdgts[pair_label]['IHC_rotation'], pair_label, IHC_shortname, _on_rotation_change)


        # Compartmentalize preview items
        pairBox = Box(children=[wdgts[pair_label]['pair_chbox']], layout=pair_layout)
        HE_paramsBox = Box(children=[wdgts[pair_label]['HE_chroma'], 
                                     wdgts[pair_label]['HE_minslider'], 
                                     wdgts[pair_label]['HE_maxslider'], 
                                     wdgts[pair_label]['HE_level']], layout=params_layout)
        HE_histBox = Box(children=[wdgts[pair_label]['HE_histout']], layout=hist_layout)
        IHC_paramsBox = Box(children=[wdgts[pair_label]['IHC_chroma'], 
                                      wdgts[pair_label]['IHC_minslider'], 
                                      wdgts[pair_label]['IHC_maxslider'], 
                                      wdgts[pair_label]['IHC_level'],
                                      wdgts[pair_label]['IHC_rotation']], layout=params_layout)
        IHC_histBox = Box(children=[wdgts[pair_label]['IHC_histout']], layout=hist_layout)
        #matchBox = wdgts[pair_label]['segment_matcher'] #20


        # Display preview window
        wdgts[pair_label]['display_box'] = Box([widgets.HTML(value='<hr>'), 
                                                pairBox,
                                                HE_paramsBox, 
                                                HE_histBox,
                                                IHC_paramsBox, 
                                                IHC_histBox,
                                                wdgts[pair_label]['segment_matcher']],
                                                layout=display_layout)
        display(wdgts[pair_label]['display_box'])
    return wdgts