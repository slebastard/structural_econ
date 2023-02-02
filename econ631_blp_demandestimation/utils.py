import numpy as np
import pandas as pd
import statsmodels.api as sm_
from statsmodels.sandbox.regression.gmm import GMM, IV2SLS

from pandas.testing import assert_index_equal

# ======================================== #
# == METHODS FOR ONE-LINE IV-REGRESSION == #
# ======================================== #

def run_2SLS(df:pd.DataFrame, instrument:list):
  Y = df["log_share"]
  X = sm_.add_constant(df[["price", "xvar"]])
  Z = sm_.add_constant(df[instrument + ["xvar"]])
  logit_2sls = IV2SLS(endog=Y,exog=X, instrument=Z).fit()
  constant, alpha, beta = logit_2sls.params
  print(logit_2sls.summary())
  return logit_2sls

def run_2SGMM(df:pd.DataFrame, instrument:list):
  Y = df["log_share"]
  X = sm_.add_constant(df[["price", "xvar"]])
  Z = sm_.add_constant(df[instrument + ["xvar"]])
  logit_2sgmm = IVGMM(endog=Y,exog=X,instrument=Z).fit();
  constant, alpha, beta = logit_2sls.params
  print(logit_2sgmm.summary())
  return logit_2sgmm


# ======================================= #
# == ELASTICITIES & DEMAND DERIVATIVES == #
# ======================================= #

def get_logit_elasticities(mktid: int) -> pd.DataFrame:
    constant, alpha, beta = logit_2sgmm.params
    df_mkt = df[df.index.get_level_values('mktid')==mktid].droplevel('mktid')
    N = df_mkt["price"].shape[0]
    print(N)
    mkt_elst = -alpha*np.dot(np.ones((N,1)), (df_mkt["price"]*df_mkt["share"]).to_numpy().reshape((1,N))) + alpha*np.diag(df_mkt["price"])
    return pd.DataFrame(mkt_elst, index=df_mkt.index, columns=df_mkt.index)

def get_demand_derivatives_logit(mktid: int) -> pd.DataFrame:
    constant, alpha, beta = logit_2sgmm.params
    df_mkt = df[df.index.get_level_values('mktid')==mktid].droplevel('mktid')
    N = df_mkt["share"].shape[0]
    #dqdp = np.dot(df_mkt["share"],(1-df_mkt["share"])) - alpha*np.ones((N,1)) + alpha*np.diag(df_mkt["share"]*df_mkt["share"])
    mkt_elst = -alpha*np.dot(np.ones((N,1)), (df_mkt["share"]*(1-df_mkt["share"])).to_numpy().reshape((1,N))) + alpha*np.diag(df_mkt["share"])
    return pd.DataFrame(mkt_elst, index=df_mkt.index, columns=df_mkt.index)