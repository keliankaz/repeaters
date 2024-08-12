import string
import json
from pathlib import Path
from typing import Optional
import os
import matplotlib.pyplot as plt

def savefig(
    fig_name, 
    save_fig_bool: bool = True, 
    save_dir: Path = Path('./Figures'), 
    metadata: Optional[dict] = None,
    figure_kwargs: Optional[dict] = None,
):
    """Saves current figures in dedicated repository.
    
    Input:
        save_fig_bool: whether or not to save the figure. e.g. Set to False if doing exploratory work.
        save_dir: Figures are saved to this directory
        versioned_directory: If set to True a new directory is created each time appending the version number
        metadata: analysis metadata is saved as JSON file in the same directory as figures
        figure_kwargs: passed on to matplotlib.pyplot.savefig as key word arguments. 
        
    By default passes: 
    ..., transparent=True, bbox_inches='tight') 
    To matplotlib.pyplot.savefig
    
    Sample usage:
    
    >> import matplotlib.pyplot as plt
    >> from pathlib import Path
    >> results_path = Path("./Results")
    >> x = [1,2,3]
    >> y = [2,3,2]
    >> metadata = dict(x=x,y=y,plottype='scatter')
    >> Pa
    >> fig, ax = plt.subplots(dpi=300,figsize=(3,4))
    >> plt.scatter(x,y)
    >> savefig('scatter_plot',savedir=results_path, metadata=metadata)
    
    """
    if save_fig_bool:
        if not save_dir.exists():
            os.makedirs(save_dir)
            
        if metadata is not None:
            save_path = save_dir/'metadata.json'
            save_path.write_text(json.dumps(metadata)),
            
        default_figure_kwargs = dict(
            transparent=True, bbox_inches='tight',
        )
        
        if figure_kwargs:
            default_figure_kwargs.update(figure_kwargs)    
            
        plt.savefig(save_dir/f'{fig_name}.pdf',**default_figure_kwargs)

def add_lettering(AX, position=[0.008, 0.925]):
    if type(position[0]) is float:
        position = len(AX) * [position]
    else:
        if len(position) != len(AX): 
            raise ValueError
    
    for (n, ax), pos in zip(enumerate(AX), position):
        ax.text(*pos, string.ascii_uppercase[n], transform=ax.transAxes, weight='bold')