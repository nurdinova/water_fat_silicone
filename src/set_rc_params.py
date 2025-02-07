# plot style
from matplotlib import font_manager
import matplotlib.pyplot as plt

def set_rc_params(rcParams):
    font_manager._load_fontmanager(try_read_cache=False)
    rcParams['figure.max_open_warning'] = False
    rcParams['figure.figsize'] = (6,4)
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Arial']
    rcParams['image.cmap'] = 'gray'
    rcParams['axes.linewidth'] = 2
    rcParams['font.size'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['grid.linewidth'] = 2
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'
    rcParams['figure.titleweight'] = 'bold'
    rcParams['figure.dpi'] = 100

    # for pdf export fonts
    rcParams["pdf.use14corefonts"] = True
    # trigger core fonts for PS backend
    rcParams["ps.useafm"] = True
    
    return rcParams