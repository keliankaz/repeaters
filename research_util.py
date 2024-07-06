import string

def add_lettering(AX, position=[0.008, 0.925]):
    for n, ax in enumerate(AX):
        ax.text(*position, string.ascii_uppercase[n], transform=ax.transAxes, weight='bold')