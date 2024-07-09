import string

def add_lettering(AX, position=[0.008, 0.925]):
    if type(position[0]) is float:
        position = len(AX) * [position]
    else:
        if len(position) != len(AX): 
            raise ValueError
    
    for (n, ax), pos in zip(enumerate(AX), position):
        ax.text(*pos, string.ascii_uppercase[n], transform=ax.transAxes, weight='bold')