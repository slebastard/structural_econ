import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, multinomial, norm

us_census = pd.read_csv("census-ageincome-joint-small.csv", index_col=0).astype(int)
census_density = (1./us_census.sum().sum())*us_census
n_age, n_inc = us_census.shape

def draw_from_census(n:int, normalize:bool=True) -> pd.DataFrame:
    census_distr = multinomial(n=1, p=census_density.to_numpy().flatten())
    census_draws = census_distr.rvs(n)
    dem_ids = np.array([np.argwhere(census_draws[k,:].reshape(n_age, n_inc))[0] for k in range(census_draws.shape[0])]).astype(float)
    if normalize:
        dem_ids[:,0] = dem_ids[:,0]/(n_age-1)
        dem_ids[:,1] = dem_ids[:,1]/(n_inc-1)
    return pd.DataFrame.from_dict({'age': dem_ids[:,0], 'income': dem_ids[:,1]})

def draw_from_normal(n: int) -> pd.DataFrame:
    return pd.DataFrame.from_dict({'age': norm.rvs(loc=0, scale=1, size=n), 'income': norm.rvs(loc=0, scale=1, size=n)})

def simulate_consumers(n:int=500, scale:float=1, sample_demographics:str='norm') -> pd.DataFrame:
    cons_df = pd.Series(data=gumbel_r.rvs(scale=scale, size=n), name='eps')
    if sample_demographics=='norm':
        demographics = draw_from_normal(n)
    elif sample_demographics=='census':
        demographics = draw_from_census(n)
    cons_df = pd.concat([cons_df, demographics], axis=1)
    return cons_df