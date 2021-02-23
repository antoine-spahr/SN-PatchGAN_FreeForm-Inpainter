"""
author: Antoine Spahr

date : 27.10.2020

----------

TO DO :
"""
import functools
import os
import json

class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed like attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        """
        Build a AttrDict from dict like this : AttrDict.from_nested_dicts(dict)
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_json_path(path):
        """ Construct nested AttrDicts from a json. """
        assert os.path.isfile(path), f'Path {path} does not exist.'
        with open(path, 'r') as fn:
            data = json.load(fn)
        return AttrDict.from_nested_dicts(data)

    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key])
                                for key in data})

def print_progessbar(N, Max, Name='', Size=10, end_char='', erase=False):
    """
    Print a progress bar. To be used in a for-loop and called at each iteration
    with the iteration number and the max number of iteration.
    ------------
    INPUT
        |---- N (int) the iteration current number
        |---- Max (int) the total number of iteration
        |---- Name (str) an optional name for the progress bar
        |---- Size (int) the size of the progress bar
        |---- end_char (str) the print end parameter to used in the end of the
        |                    progress bar (default is '')
        |---- erase (bool) whether to erase the progress bar when 100% is reached.
    OUTPUT
        |---- None
    """
    print(f'{Name} {N+1:04d}/{Max:04d}'.ljust(len(Name) + 12) \
        + f'|{"█"*int(Size*(N+1)/Max)}'.ljust(Size+1) + f'| {(N+1)/Max:.1%}'.ljust(6), \
        end='\r')

    if N+1 == Max:
        if erase:
            print(' '.ljust(len(Name) + Size + 40), end='\r')
        else:
            print('')
