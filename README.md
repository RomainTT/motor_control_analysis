# Motor control analysis

This repository contains some tools to perform the analysis of a motor control.
The goal is to standardize the way data is processed and information is displayed.

## Content description

### ./template_notebook.ipynb

This is the template of any notebook used to perform and present the analysis of 
a motor control in a particular context. This template ensures that every analysis
will follow the same structure and representation, making it easier to switch from
a particular study to another.

If you want to make a new analysis, just copy-paste this template and fill it 
as it is instructed inside the template.

### ./utils

This directory contains some Python modules used by the notebooks. The following
modules can be found:
* `data`: contains functions to get/browse raw data and to save post-processed data.
* `plot`: contains functions to generate and display plots and synthesis pages of 
  motor control KPIs.
* `image`: contains functions to perform image manipulations. Only used by the `plot`
  module, not directly by notebooks.
* `signal.generic`: contains some functions to perform signal processing, not 
  specifically for motor control.
* `signal.generic`: contains some functions to perform signal processing specific
  for motor control, including the computation of control KPIs.

### ./resources

The directory `resources` contains some pictures used in the notebooks. Here are
some notes about these pictures:

* Most of the pictures have two files: one `.odg` file to edit them with 
  *LibreOffice Draw*, and one `.png` to use them in the notebook. `.png` can be
  exported directly from *LibreOffice Draw* in most of the cases.
* The picture `template_synthesis` must not be exported with *LibreOffice Draw*
  because the quality would be too low. To convert the `.odg` into a `.png`,
  the `.odg` file must be exported to a `.svg` file and then 
  *Inkscape* is used with the following command line:  
  `inkscape -z -e template_synthesis.png -w 2000 -b 'white' template_synthesis.svg`.  
  As it can be seen in the command line, a width of 2000 pixels has been chosen
  for this particular image.


