"""Diacritic / non-ASCII letter folding, consistent for indexing and querying."""
import unicodedata
_MAP = str.maketrans({
    'Ł':'L','ł':'l','Ø':'O','ø':'o','Æ':'AE','æ':'ae','ß':'ss','Đ':'D','đ':'d',
    'Þ':'Th','þ':'th','ı':'i','İ':'I','Œ':'OE','œ':'oe','Ð':'D','ð':'d','Ħ':'H','ħ':'h',
    'Ŀ':'L','ŀ':'l','ĸ':'k','Ŋ':'N','ŋ':'n','Ŧ':'T','ŧ':'t',
})
def fold(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.translate(_MAP)
