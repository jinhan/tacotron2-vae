""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text 
that has been run through Unidecode. 
For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

from text import cmudict

include_capital = False
include_arpabet = False
include_label = False

_pad        = '_' # must be put in first to correspond to symbol 0 (i.e. pad with 0s)
_punctuation = '!\'(),.:;? '
_special = '-'
_end = '~'

# Export all basic symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_end)

if include_capital:
  _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
else:
  _letters = 'abcdefghijklmnopqrstuvwxyz'
symbols += list(_letters)

if include_arpabet:
    # Prepend "@" to ARPAbet symbols to ensure uniqueness
    # (some are the same as uppercase letters)
    _arpabet = ['@' + s for s in cmudict.valid_symbols]
    symbols += _arpabet

if include_label:
    # prepare labels for emotions
    _labels = ['@' + s for s in '0123']
    symbols += _labels

print('symbol including capital: {}'.format(include_capital))
print('symbol including arpabet: {}'.format(include_arpabet))
print('symbol including label: {}'.format(include_label))
print('# of symbols: {}'.format(len(symbols)))
