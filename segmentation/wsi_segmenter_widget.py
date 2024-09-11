# I/O Handling
from __future__ import print_function
import csv,glob,re
import os,sys

# Widgets
import ipywidgets as widgets
from ipywidgets import Text, Label
from ipywidgets import Layout, HBox, VBox, Box
from ipywidgets import Button, Accordion

# Custom functions
from preview_segmentations import preview_segmentations
from widget_utils import collect_HE_wparams, collect_IHC_wparams, collect_matcher_wparams
#from load_widget_parameters import *


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


class SegmentationWidget(object):
    """IpyWidget for interactive preview/editing of WSI segmentations"""
    def __init__(self, default_dir='.', load_params=None):
        
        # Initialize Previews
        self.preview_widgets = {}

        # Widget Layouts
        self.file_selection_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='90%')
        self.preview_widgets_layout = Layout(display='flex', flex_flow='row', align_items='flex-start', 
                                             justify_content='flex-start', widgth='22em')
        self.save_params_layout = Layout(display='flex', flex_flow='column', width='auto', visibility='hidden')

        # Load Parameters Widgets
        self.load_wparams_btn = Button(description="Load Widget Parameters File", layout=widgets.Layout(width="17em"))
        self.load_wparams_btn.on_click(self.load_widget_params)
        self.load_wparams_tBox = Text(value=load_params, placeholder="Widget parameters filepath",
                                      layout=Layout(flex='2 1 auto', width='auto', justify_content='space-between'))
        self.load_wparams_Box = Box(children=[self.load_wparams_btn, self.load_wparams_tBox], layout=self.file_selection_layout)

        # Preview Widgets
        #    Directory selection
        self.imDir_label = HBox([Label('Directory:', 
                                 layout=Layout(flex='1 1 auto', width='auto', justify_content='space-between'))])
        self.imDir_tBox = Text(value=default_dir, placeholder="Path to directory containing HE+IHC files",
                               layout=Layout(flex='3 1 auto'), width='auto')
        self.imDir_Box = Box(children=[self.imDir_label, self.imDir_tBox], layout=self.file_selection_layout)
        #    File-pattern selection
        self.imFn_label = HBox([Label('File Pattern:', layout=Layout(flex='1 1 auto', width='auto'))])
        self.imFn_tBox = Text(value='H13-11*.[svstiff]*', placeholder="File pattern (eg. 12-1a*.svs)",
                              layout=Layout(flex='2 1 auto', width='auto', justify_content='space-between'))
        self.imFn_Box = Box(children=[self.imFn_label, self.imFn_tBox], layout=self.file_selection_layout)
        #    Preview image-levels
        self.preview_heLevel = widgets.IntText(value=2, description='HE Level', disabled=False, layout=Layout(width='8em'))
        self.preview_ihcLevel = widgets.IntText(value=2, description='IHC Level', disabled=False, layout=Layout(width='8em'))
        #    Preview segments
        self.preview_seg_btn = Button(description="Preview Segmentations", layout=widgets.Layout(width='13em'))
        self.preview_seg_btn.on_click(self._preview_segments)
        self.preview_Box = Box(children=[self.preview_heLevel, self.preview_ihcLevel, self.preview_seg_btn], layout=self.preview_widgets_layout)

        # Save Widget Parameters
        #self.save_widget_params = self._create_wdgt_parameter_options()

        # Save Segment Parameters
        self.save_segment_params = self._create_segment_parameter_options()
        
        # Output
        self.std_out = widgets.Output()

        # Display
        self.separator = widgets.HTML(value="<hr>")
        self.main_layout = Box([self.load_wparams_Box,
                                self.separator,
                                self.imDir_Box, 
                                self.imFn_Box,
                                self.preview_Box,
                                self.separator,
                                #self.save_widget_params,
                                self.save_segment_params,
                                self.std_out], 
                                layout=Layout(display='flex', flex_flow='column', align_items='stretch'))
        display(self.main_layout)




    '''
    #############################################################################################################
    #                                          Methods/Attributes                                                
    #############################################################################################################
    '''

    def load_widget_params(self, btn):
        """Button function: Loads widget parameters as set by wparams file"""
        pass
    #    self._close_preview()
    #    self._show_widget_params()
    #    self.std_out.clear_output()
    #    self.preview_widgets = {}
    #    widget_params_f = self.load_wparams_tBox.value
    #    print('Loading widget parameters from file: {}'.format(widget_params_f))
    #    self.preview_widgets = load_widget_parameters(widget_params_f, wdgts=self.preview_widgets)



    def select_all_previews(self, btn):
        """Widget parameter option: Activates checkbox for all HE-IHC previews"""
        for w in self.preview_widgets.values():
            w['pair_chbox'].value = True



    def deselect_all_previews(self, btn):
        """Widget parameter option: Deactivates checkbox for all HE-IHC previews"""
        for w in self.preview_widgets.values():
            w['pair_chbox'].value = False



    def _create_segment_parameter_options(self):
        """Creates options for saving segment parameters related to all activated (checkboxed) previews"""
        
        # Segment Parameters Save-Options
        #    File-path selection
        sparams_label = HBox([Label('Segment Params Outfile:', layout=Layout(flex='1 1 auto', width='auto'))])
        self.sparams_tBox = Text(value='/scratch/disk1/ggaskins/param_files/segmentation_params/stanford40x/seg-params_{}.csv'
                                       .format(self.imFn_tBox.value.split('.')[0]), 
                                 placeholder="Filepath to write segment parameters", 
                                 layout=Layout(flex='2 1 auto', width='auto', justify_content='space-between'))
        #    Save segment-params button
        write_sparams_btn = Button(description="Write Segment Params", layout=Layout(width='16em'))
        write_sparams_btn.on_click(self.save_segment_params)
        
        # Segment Save-Options
        #    HE-Mask Directory selection
        segments_outdir_label = HBox([Label('Segment-Masks Outdir:', layout=Layout(flex='1 1 auto', width='auto'))])
        self.segments_outdir_tBox = Text(value='/scratch/disk1/ggaskins/masked_images/stanford40x', 
                                         placeholder="Directory to store masked segments", 
                                         layout=Layout(flex='2 1 auto', width='auto', justify_content='space-between'))
        #    Segment all button
        segment_all_btn = Button(description="Segment Sections", layout=Layout(width="16em"))
        segment_all_btn.on_click(self.segment_sections)

        # Layout Design
        sparams_filepath_Box = Box(children=[sparams_label, self.sparams_tBox], layout=self.file_selection_layout)
        segments_outdir_Box = Box(children=[segments_outdir_label, self.segments_outdir_tBox], layout=self.file_selection_layout)
        sparams_buttons_Box = HBox(children=[write_sparams_btn, segment_all_btn])
        accordion = Accordion(children=[VBox([sparams_filepath_Box, segments_outdir_Box, sparams_buttons_Box])])
        accordion.set_title(0, 'Save Parameters+Segments to File')
        accordion.selected_index = None
        sparams_options = Box(children=[accordion], layout=self.save_params_layout)
        return sparams_options



    def _show_segment_params(self):
        if self.save_segment_params:      
            self.save_segment_params.layout.visibility = 'visible'



    def _preview_segments(self, btn):
        """Button function: Generates WSI segmentation previews"""
        self._close_preview()
        self._show_segment_params()
        self.std_out.clear_output()
        self.preview_widgets = preview_segmentations(imDir=self.imDir_tBox.value, imFn_pattern=self.imFn_tBox.value, 
                                                     wdgts=self.preview_widgets, heLevel=self.preview_heLevel.value, 
                                                     ihcLevel=self.preview_ihcLevel.value)
        if not self.preview_widgets:
            self._close_preview()
            with self.std_out:
                fn = os.path.join(self.imDir_tBox.value, self.imFn_tBox.value)
                print("Unable to segment images matching: {}".format(fn))
        return



    def save_segment_params(self, btn):
        """Button function: Saves widget parameters to csv file."""

        # Prep-area
        outfn = self.sparams_tBox.value
        print('Saving widget parameters to: {}'.format(outfn))
        header = ['HE_wsi', 'HE_chroma', 'HE_minThresh', 'HE_maxThresh',
                  'IHC_wsi', 'IHC_chroma', 'IHC_minThresh' , 'IHC_maxThresh', 'IHC_rotate',
                  'segment_matches']
        
        # Open writer object
        with open(outfn, 'w') as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            
            # Loop through previews, if preivew is checked, write widget parameters
            for wdict in self.preview_widgets.values():
                if wdict['pair_chbox'].value:
                    
                    # HE parameters
                    HE_wparams = collect_HE_wparams(wdict)
                    HE, HE_chroma, HE_min, HE_max = HE_wparams

                    # IHC parameters
                    IHC_wparams = collect_IHC_wparams(wdict)
                    IHC, IHC_chroma, IHC_min, IHC_max, IHC_rotate = IHC_wparams
                    
                    # Construct string detailing HE-IHC segment matches
                    segment_pairs = []
                    match_labels, match_children = collect_matcher_wparams(wdict)
                    #print('match_labels: {}'.format(match_labels))
                    #print('match_children: {}'.format(match_children))
                    for idx,label in enumerate(match_labels[:-1]):
                        segment_pairs.append('+'.join([label, match_children[idx].value]))
                    match_string = ';'.join(segment_pairs)
                    
                    # Collate parameters and write to file
                    params = [HE, HE_chroma, HE_min, HE_max,
                              IHC, IHC_chroma, IHC_min, IHC_max, IHC_rotate,
                              match_string]
                    writer.writerow(params)

    def segment_sections(self, btn):
        """Segments, extracts, and masks sections from full WSI of all checked preview images"""

        # Prep-area
        WSI_to_segments = {}
        basename = os.path.basename(self.sparams_tBox.value)
        outfn = os.path.join(self.segments_outdir_tBox.value, basename.replace('.csv', '.mask.csv'))
        header = ['HE_mask', 'IHC_mask']
        print('Saving segment masks to: {}'.format(outfn))

        # Setup outfile writer
        with open(outfn, 'w') as fo:
            writer = csv.writer(fo)
            writer.writerow(header)

            # Loop through checked previews gather HE-IHC pairings
            for wdict in self.preview_widgets.values():
                if wdict['pair_chbox'].value:
                    
                    # HE parameters
                    HE_wparams = collect_HE_wparams(wdict)
                    HE, HE_chroma, HE_min, HE_max, HE_dilate = HE_wparams

                    # IHC parameters
                    IHC_wparams = collect_IHC_wparams(wdict)
                    IHC, IHC_chroma, IHC_min, IHC_max, IHC_dilate, IHC_rotate = IHC_wparams
                    
                    # Construct string detailing HE-IHC segment matches
                    segment_pairs = []
                    match_labels, match_children = collect_matcher_wparams(wdict)

                    # Get HE-IHC pair indexs
                    HE_segments = []
                    IHC_segments = []
                    for idx,label in enumerate(match_labels[:-1]):
                        HE_segidx = int(label.split('.')[1])
                        HE_segments.append(HE_segidx)
                        IHC_segidx = int(match_children[idx].value.split('.')[1])
                        IHC_segments.append(IHC_segidx)
                    WSI_to_segments['+'.join([HE, IHC])] = list(zip(HE_segments, IHC_segments))
        print(WSI_to_segments)



    def _close_preview(self):
        """Resets preview display window"""
        if self.save_segment_params:
            self.save_segment_params.layout.visibility = 'hidden'
        if self.preview_widgets:
            for w in self.preview_widgets.values():
                if w['display_box']:
                    w['display_box'].close()
                w['display_box'] = None
            self.preview_widgets = {}