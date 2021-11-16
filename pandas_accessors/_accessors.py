import functools
import warnings

from pandas import DataFrame, Index, Series
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor, register_index_accessor


def silently(register):
    """Provide accessor to pandas register functions without warning."""

    def decorator(name):
        
        def register_accessor(accessor):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                register(name)(accessor)
                
        return register_accessor
            
    return decorator

register_dataframe_accessor_silently = silently(pd.api.extensions.register_dataframe_accessor)


@register_dataframe_accessor('axis')
class AxisAccessor:
    def __init__(self, df: pd.Series):
        self.df = df

    def __call__(self, axis):
        df = self.df
        return df.columns if axis == 1 else df.index

    
@register_dataframe_accessor_silently('mul')
class Mul:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, other=None, level=0, axis=0, fill_value=None, *args, **kwargs):
        df = self.df
        if other:
            out = df.mul(other, *args, **kwargs)
        else:
            out = functools.reduce(
                lambda x, y: df[x].mul(df[y], level=level, axis=axis, fill_value=fill_value),
                df.axis(axis).get_level_values(level).unique()
            )
        return out
    

@register_dataframe_accessor('with_index')
@register_series_accessor('with_index')
class WithIndex:
    def __init__(self, sdf):
        self.sdf = sdf
        self.index = sdf.index.copy()
    
    def __call__(self, f):
        sdf = self.sdf
        sdf.index = f(sdf.index).copy()
        return sdf
    
    def __enter__(self, f, *args, **kwargs):
        index = self.index.copy()
        sdf.index = f(index)
        
    def __exit__(self, *args, **kwargs):
        sdf.index = self.index
    
    
@register_dataframe_accessor('joinindex')
@register_series_accessor('joinindex')
class JoinIndex:
    def __init__(self, sdf):
        self.sdf = sdf
    
    def __call__(self, index):
        sdf = self.sdf
        sdf.index = sdf.index.join(index).sort_values(
            key=sdf.index.get_indexer)

        return sdf
    
    
@register_dataframe_accessor('joincolumns')
@register_series_accessor('joincolumns')
class JoinColumns:
    def __init__(self, sdf):
        self.sdf = sdf
    
    def __call__(self, index):
        sdf = self.sdf
        sdf.columns = sdf.columns.join(index).sort_values(
            key=sdf.columns.get_indexer)

        return sdf