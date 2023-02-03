import numpy as np
import pandas as pd
import statsmodels.api as sm_

# ========================== #
# == BUILDING INSTRUMENTS == #
# ========================== #

def HausmanIV(df: pd.DataFrame) -> pd.DataFrame:
    hausman_ivs = {}
    for mktid, df_mkt in df.groupby(by="mktid"):
        df_othermkts = df[~df.index.get_level_values('mktid').isin([mktid])]
        hausman_priceavg = df_othermkts.groupby(by=["firmid", "prodid"])["price"].mean()
        hausman_priceavg = pd.concat([hausman_priceavg], keys=[mktid], names=['mktid'])
        hausman_ivs[mktid] = hausman_priceavg
    hausman_ivs_df = pd.concat(hausman_ivs.values(), keys=hausman_ivs.keys()).rename('hausman_iv').droplevel(0)
    return pd.merge(left=df, right=hausman_ivs_df, on=df.index.names)


def BLPIV(df):
  df3 = df.reset_index()
  T = len(df3.mktid.unique())
  J = len(df3.prodid.unique())
  N = len(df3.firmid.unique())
  D1 = np.zeros((N,T))
  D2 = np.zeros((N,T))
  D3 = np.zeros((N,J,T))
  D4 = np.zeros((N,J,T))

  for t in range(1, 21):
    for n in range(1, 5):

      # BLP1: Number of products owned by competing firm
      D1[n-1,t-1] = len(df3[(df3.firmid!=n) & (df3.mktid==t)].prodid.unique())
      
      # BLP2: Total number of products by market (co-linear in these data)
      D2[n-1,t-1] = len(df3[(df3.mktid==t)].prodid.unique())
  
  for t in range(1, 21):
    for n in range(1, 5):
      for j in range(1, 21):

        # BLP3: Sum of attributes in the market for each product produced by firm excluding product j
        D3[n-1,j-1,t-1] = df3[(df3.firmid==n) & (df3.mktid==t) & (df3.prodid!=j)].xvar.sum()
        
        # BLP4: Sum of attributes of products owned by competing firm
        D4[n-1,j-1,t-1] = df3[(df3.firmid!=n) & (df3.mktid==t) & (df3.prodid!=j)].xvar.sum()

  # write the D matrices (4,20) back into the dataframe indexed by firmid, mktid
  df3['blp1']=0
  df3['blp2']=0
  df3['blp3']=0
  df3['blp4']=0

  for t in range(1, 21):
    for n in range(1, 5):

      df3.loc[(df3.mktid==t) & (df3.firmid==n),'blp1'] = D1[n-1,t-1]
      df3.loc[(df3.mktid==t) & (df3.firmid==n),'blp2'] = D2[n-1,t-1]

  for t in range(1, 21):
    for n in range(1, 5):
      for j in range(1, 21):

        df3.loc[(df3.mktid==t) & (df3.firmid==n) & (df3.prodid==j),'blp3'] = D3[n-1,j-1,t-1]
        df3.loc[(df3.mktid==t) & (df3.firmid==n) & (df3.prodid==j),'blp4'] = D4[n-1,j-1,t-1]

  df3 = df3.set_index(['mktid', 'prodid', 'firmid'])
  return df3


def DifferentiationIV(df):
    df2 = df.reset_index()
    T = len(df2.mktid.unique())
    N = len(df2.firmid.unique())
    J = len(df2.prodid.unique())
    D = np.zeros((J,J,T))

    for j in range(J):
        for k in range(J):
            for t in range(T):
                try:
                    D[j,k,t] = df2[(df2.prodid==j) & (df2.mktid==t)].xvar.item() - df2[(df2.prodid==k) & (df2.mktid==t)].xvar.item() 
                except: 
                    D[j,k,t] = 0

    D2 = D**2

    df2['prod_char_dist']=0
    for t in range(T):
        for j in range(J):
            temp = 0
            for k in range(J):
                if k != j:
                    temp += D2[j, k, t]**2
            df2.loc[(df2.mktid==t) & (df2.prodid==j),'prod_char_dist'] = temp

    df2['prod_band0']=0
    df2['prod_band1']=0
    df2['prod_band2']=0
    for t in range(T):
        for j in range(J):
            q_2 = np.quantile(np.abs(D[j,:,t]), 0.5)
            q_5 = np.quantile(np.abs(D[j,:,t]), 0.75)
            q_7 = np.quantile(np.abs(D[j,:,t]), 0.95)
            #print(q_2, q_5, q_7)
            for (i,q) in enumerate([q_2, q_5, q_7]):
                temp = 0
                for k in range(J):
                    if k != j:
                        if np.abs(D2[j, k, t]) < q:
                            temp += 1
                name = 'prod_band' + str(i)
                df2.loc[(df2.mktid==t) & (df2.prodid==j), name] = temp
                
    print(np.mean(df2['prod_char_dist']), np.mean(df2.prod_band0), np.mean(df2.prod_band1), np.mean(df2.prod_band2))
    df2 = df2.set_index(['mktid', 'prodid', 'firmid'])
    df2 = df2.fillna(0)
    df2.rename(columns={'prod_char_dist': 'diff_iv', 'prod_band0': 'diff_iv_prodband0', 'prod_band1': 'diff_iv_prodband1', 'prod_band2': 'diff_iv_prodband2'}, inplace=True)
    return df2


def normalize_instr(df:pd.DataFrame, instruments:list):
    df_normed = df.copy()
    for instr in instruments:
        df_normed[instr] = (df_normed[instr] - df_normed[instr].min()) / (df_normed[instr].max() - df_normed[instr].min())
    return df_normed