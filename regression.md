## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important


```python
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols as sm_ols
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col
```


```python
housing = pd.read_csv('input_data2/housing_train.csv')
housing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcel</th>
      <th>v_MS_SubClass</th>
      <th>v_MS_Zoning</th>
      <th>v_Lot_Frontage</th>
      <th>v_Lot_Area</th>
      <th>v_Street</th>
      <th>v_Alley</th>
      <th>v_Lot_Shape</th>
      <th>v_Land_Contour</th>
      <th>v_Utilities</th>
      <th>...</th>
      <th>v_Pool_Area</th>
      <th>v_Pool_QC</th>
      <th>v_Fence</th>
      <th>v_Misc_Feature</th>
      <th>v_Misc_Val</th>
      <th>v_Mo_Sold</th>
      <th>v_Yr_Sold</th>
      <th>v_Sale_Type</th>
      <th>v_Sale_Condition</th>
      <th>v_SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1056_528110080</td>
      <td>20</td>
      <td>RL</td>
      <td>107.0</td>
      <td>13891</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>372402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1055_528108150</td>
      <td>20</td>
      <td>RL</td>
      <td>98.0</td>
      <td>12704</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>317500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1053_528104050</td>
      <td>20</td>
      <td>RL</td>
      <td>114.0</td>
      <td>14803</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>385000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2213_909275160</td>
      <td>20</td>
      <td>RL</td>
      <td>126.0</td>
      <td>13108</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>153500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1051_528102030</td>
      <td>20</td>
      <td>RL</td>
      <td>96.0</td>
      <td>12444</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>394617</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>2524_534125210</td>
      <td>190</td>
      <td>RL</td>
      <td>79.0</td>
      <td>13110</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>146500</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>2846_909131125</td>
      <td>190</td>
      <td>RH</td>
      <td>NaN</td>
      <td>7082</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>2605_535382020</td>
      <td>190</td>
      <td>RL</td>
      <td>60.0</td>
      <td>10800</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>ConLD</td>
      <td>Normal</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>1516_909101180</td>
      <td>190</td>
      <td>RL</td>
      <td>55.0</td>
      <td>5687</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>135900</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>1387_905200100</td>
      <td>190</td>
      <td>RL</td>
      <td>60.0</td>
      <td>12900</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>95541</td>
    </tr>
  </tbody>
</table>
<p>1941 rows × 81 columns</p>
</div>



## Takeaways

- Unit of obs is housing units (should be 1941 obs)
- Time span covered is 2006-2008
- There are some outliers, missing values for certain variables

## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? I won't be shocked if someone beats that!
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
reg1 = sm_ols("v_SalePrice ~ v_Lot_Area", data=housing).fit()
reg2 = sm_ols("v_SalePrice ~ np.log(v_Lot_Area)", data=housing).fit()
reg3 = sm_ols("np.log(v_SalePrice) ~ v_Lot_Area", data=housing).fit()
reg4 = sm_ols("np.log(v_SalePrice) ~ np.log(v_Lot_Area)", data=housing).fit()
reg5 = sm_ols("np.log(v_SalePrice) ~ v_Yr_Sold", data=housing).fit()
reg6 = sm_ols("np.log(v_SalePrice) ~ v_Yr_Sold==2007 + v_Yr_Sold==2008", data=housing).fit()
reg7 = sm_ols("np.log(v_SalePrice) ~ v_Lot_Area + v_Yr_Sold + v_Lot_Config + v_Neighborhood + v_Overall_Qual", data=housing).fit()
```


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"} 

# This summary col function combines a bunch of regressions into one nice table
print('='*107)
print('                  y = Sale  Price if not specified, log(Sale Price else)')
print(summary_col(results=[reg1,reg2,reg3,reg4,reg5,reg6,reg7], # list the result obj here
                  float_format='%0.2f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3','4','5','6','7'], # these are bad names, lol. Usually, just use the y variable name
                  info_dict=info_dict,
                  regressor_order=[ 'v_SalePrice','v_Lot_Area','np.log(v_Lot_Area)','v_yr_Sold','v_Lot_Config',
                                  'v_Neighborhood','v_Overall_Qual']
                  )
     )
```

    ===========================================================================================================
                      y = Sale  Price if not specified, log(Sale Price else)
    
    ===============================================================================================
                                   1             2           3       4       5       6        7    
    -----------------------------------------------------------------------------------------------
    v_Lot_Area                2.65***                    0.00***                           0.00*** 
                              (0.23)                     (0.00)                            (0.00)  
    np.log(v_Lot_Area)                     56028.17***            0.29***                          
                                           (3315.14)              (0.02)                           
    v_Overall_Qual                                                                         0.17*** 
                                                                                           (0.00)  
    Intercept                 154789.55*** -327915.80*** 11.89*** 9.41*** 22.29   12.02*** 20.82*  
                              (2911.59)    (30221.35)    (0.01)   (0.15)  (22.94) (0.02)   (10.94) 
    v_Neighborhood[T.SawyerW]                                                              -0.00   
                                                                                           (0.05)  
    v_Neighborhood[T.NAmes]                                                                -0.08*  
                                                                                           (0.05)  
    v_Neighborhood[T.NPkVill]                                                              -0.19** 
                                                                                           (0.08)  
    v_Neighborhood[T.NWAmes]                                                               -0.00   
                                                                                           (0.05)  
    v_Neighborhood[T.NoRidge]                                                              0.27*** 
                                                                                           (0.05)  
    v_Neighborhood[T.NridgHt]                                                              0.18*** 
                                                                                           (0.05)  
    v_Neighborhood[T.OldTown]                                                              -0.21***
                                                                                           (0.05)  
    v_Neighborhood[T.SWISU]                                                                -0.13** 
                                                                                           (0.06)  
    v_Neighborhood[T.Sawyer]                                                               -0.08*  
                                                                                           (0.05)  
    v_Neighborhood[T.MeadowV]                                                              -0.26***
                                                                                           (0.06)  
    v_Neighborhood[T.Somerst]                                                              0.02    
                                                                                           (0.05)  
    v_Neighborhood[T.StoneBr]                                                              0.13**  
                                                                                           (0.05)  
    v_Neighborhood[T.Timber]                                                               0.09*   
                                                                                           (0.05)  
    v_Neighborhood[T.Veenker]                                                              0.10    
                                                                                           (0.06)  
    v_Yr_Sold                                                             -0.01            -0.00   
                                                                          (0.01)           (0.01)  
    v_Yr_Sold == 2007[T.True]                                                     0.03             
                                                                                  (0.02)           
    v_Neighborhood[T.Mitchel]                                                              -0.04   
                                                                                           (0.05)  
    v_Neighborhood[T.Landmrk]                                                              -0.19   
                                                                                           (0.20)  
    v_Neighborhood[T.BrDale]                                                               -0.42***
                                                                                           (0.06)  
    v_Neighborhood[T.IDOTRR]                                                               -0.33***
                                                                                           (0.05)  
    v_Lot_Config[T.CulDSac]                                                                0.04*   
                                                                                           (0.02)  
    v_Lot_Config[T.FR2]                                                                    -0.06*  
                                                                                           (0.03)  
    v_Lot_Config[T.FR3]                                                                    -0.01   
                                                                                           (0.06)  
    v_Lot_Config[T.Inside]                                                                 0.01    
                                                                                           (0.01)  
    v_Neighborhood[T.Blueste]                                                              -0.22** 
                                                                                           (0.10)  
    v_Yr_Sold == 2008[T.True]                                                     -0.01            
                                                                                  (0.02)           
    v_Neighborhood[T.BrkSide]                                                              -0.21***
                                                                                           (0.05)  
    v_Neighborhood[T.ClearCr]                                                              0.03    
                                                                                           (0.06)  
    v_Neighborhood[T.CollgCr]                                                              0.01    
                                                                                           (0.05)  
    v_Neighborhood[T.Crawfor]                                                              0.07    
                                                                                           (0.05)  
    v_Neighborhood[T.Edwards]                                                              -0.16***
                                                                                           (0.05)  
    v_Neighborhood[T.Gilbert]                                                              -0.02   
                                                                                           (0.05)  
    v_Neighborhood[T.Greens]                                                               -0.21** 
                                                                                           (0.10)  
    v_Neighborhood[T.GrnHill]                                                              0.30**  
                                                                                           (0.14)  
    R-squared                 0.07         0.13          0.06     0.13    0.00    0.00     0.78    
    R-squared Adj.            0.07         0.13          0.06     0.13    -0.00   0.00     0.78    
    R-squared                 0.07         0.13          0.06     0.13    0.00    0.00     0.78    
    Adj R-squared             0.07         0.13          0.06     0.13    -0.00   0.00     0.78    
    No. observations          1941         1941          1941     1941    1941    1941     1941    
    ===============================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 



```python
#1
print("N/A")
```

    N/A



```python
#2
reg2.summary()
y10k = reg2.params[0]+np.log(10000)*reg2.params[1]
y20k = reg2.params[0]+np.log(10100)*reg2.params[1]
print(f"""Reg 2: 
intercept              :  {reg2.params[0]}
beta1:                    {reg2.params[1]}
y at Lot Area == 10,000: {y10k}
y at Lot Area == 10,100: {y20k}
Going from Lot Area 10k to 10100, reg2 predicts Sale Price changes: {y20k-y10k}
""")
```

    Reg 2: 
    intercept              :  -327915.80232023844
    beta1:                    56028.16996046535
    y at Lot Area == 10,000: 188122.7134345788
    y at Lot Area == 10,100: 188680.2122627829
    Going from Lot Area 10k to 10100, reg2 predicts Sale Price changes: 557.4988282041159
    


Model 2:
Intercept --> A Sale Price of -327915.80. if log(Lot Area)=0.
"A 1% increase in Lot Area is associated with an increase of 557.50 in Sales Price."
@X=10,000, E(y) is 188122.71
10,000 to 10,100 --> Sales Price increases 557.50


```python
#3
reg3.summary()
y10k = reg3.params[0]+10000*reg3.params[1]
y20k = reg3.params[0]+10001*reg3.params[1]
print(f"""Reg 3: 
intercept              :  {reg3.params[0]}
beta1:                    {reg3.params[1]}
log(y) at Lot Area == 10,000: {y10k}
log(y) at Lot Area == 10,001: {y20k}
Going from Lot Area 10k to 10,001, reg3 predicts log(Sale Price) changes: {y20k-y10k}
""")
```

    Reg 3: 
    intercept              :  11.89407251466273
    beta1:                    1.3092338465836551e-05
    log(y) at Lot Area == 10,000: 12.024995899321095
    log(y) at Lot Area == 10,001: 12.025008991659561
    Going from Lot Area 10k to 10,001, reg3 predicts log(Sale Price) changes: 1.3092338466691444e-05
    


Model 3:
"A 1 unit increase in Lot Area is associated with a PROPORTIONAL increase of 0.0013% in Sales Price."
Lot Area 10000 to 10001 --> Sales Price increase 0.0013%

#### #4
I think Model 2 best explains the data because it shows how a proportional increase in lot area affects sales prices which is exactly what we are looking for in this scenario.


```python
#5
reg5.summary()
y2006 = reg5.params[0]+2006*reg5.params[1]
y2007 = reg5.params[0]+2007*reg5.params[1]
print(f"""Reg 5: 
intercept              :  {reg5.params[0]}
beta1:                    {reg5.params[1]}
y at Year Sold == 2006: {y2006}
y at Year Sold == 2007: {y2007}
Going from Year Sold 2006 to 2007, reg5 predicts Sale Price changes: {y2007-y2006}
""")
```

    Reg 5: 
    intercept              :  22.293213132062135
    beta1:                    -0.005114348195977281
    y at Year Sold == 2006: 12.03383065093171
    y at Year Sold == 2007: 12.028716302735733
    Going from Year Sold 2006 to 2007, reg5 predicts Sale Price changes: -0.005114348195977669
    


Model 5: "A 1 year increase in Year Sold is associated with a PROPORTIONAL increase of 0.51% in interest rates." Year Sold 2006 to 2007 --> Sales Price increase 0.51%

#### #6
Alpha is the intercept so when the dependent variable is equal to zero alpha is equal to the independent variable. 


```python
#7
reg6.summary()
y2007 = reg6.params[0]+2007*reg5.params[1]
y2008 = reg6.params[0]+2008*reg5.params[1]
print(f"""Reg 6: 
intercept              :  {reg6.params[0]}
beta1:                    {reg6.params[1]}
y at Year Sold == 2007: {y2007}
y at Year Sold == 2008: {y2008}
Going from Year Sold 2007 to 2008, reg6 predicts Sale Price changes: {y2008-y2007}
""")
```

    Reg 6: 
    intercept              :  12.022869210751955
    beta1:                    0.02559031997164936
    y at Year Sold == 2007: 1.758372381425552
    y at Year Sold == 2008: 1.7532580332295744
    Going from Year Sold 2007 to 2008, reg6 predicts Sale Price changes: -0.005114348195977669
    


Model 6: "A 1 year increase in Year Sold is associated with a PROPORTIONAL increase of 0.51% in interest rates." Year Sold 2007 to 2008 --> Sales Price increase 0.51%

#### #8
Unfortunately mine is not so I can't really comment with numbers but if I had to guess I would say it is because housing prices climbed so much in 2008 so buying a house in 2007 versus 2008 is highly correlated with a higher price in 2008.

#### #9
The variables I included in my Model 7 were v_Lot_Area, v_Yr_Sold, v_Lot_Config, v_Neighborhood, and v_Overall_Qual

#### #10
0.78

#### #11
Yes because buying a house in 2008 as opposed to 2007 clearly correlate to higher sales prices

#### #12
Probably not because buying a house in 2007 vs 2006 did not necessarily correlate to higher prices.
