---
layout: wide_default
---    

# Andrew Schachter Midterm Project

## Summary

(brief, max 2 paragraphs)

Summarize your question, what you did, and your findings. You can model this on the abstracts in the literature folder.

## Data

### What’s the sample?
The sample is the S&P 500 firms from https://en.wikipedia.org/wiki/List_of_S%26P_500_companies. This sample includes ticker symbols and ID numbers for each firm, as well as other data that was not used in this project.

### How are the return variables built and modified?

The return variables were built using the crsp dataset. Using the "date" variable from t=filing date to t+2 as well as t+3 to t+10, the data should be able to see returns over time. However, instead what I had to do was merge the crsp returns with the dataset I already had. The crsp dataset had a date variable and a ticker variable, so I had to change the variable names for the merge. Once I merged the dataset, the crsp return variable was loaded into the dataframe and subsequent csv file.

### How are the sentiment variables built and modified?

The sentiment variables were built using the Loughran and McDonald study master list as well as the machine learning approach in the Journal of Financial Economics. The variables were created from lists of sentiment words that were given positive and negative sentiments. Separate positive and negative sentiment lists were then made to create the sentiment variables.

### Why did you choose the three topics you did for the “contextual sentiment” measures?

The three topics I chose for the "contextual sentiment" measures were Porter's Five Forces, risk, and countries. I chose Porter's Five Forces because I felt as a general business topic that is highly discussed in the business world, words relating to the five forces would be well-represented in 10-K's. I chose risk because it is the most common type of business assessment and return is most often associated with risk. I chose countries because I thought it would be interesting to see if there were positive or negative associations with different countries and whether those positive/negative sentiments would vary based on the country as well as how different countries would affect returns. 

### Show and discuss summary stats of your final analysis sample



### Do your “contextual sentiment” measures pass some basic smell tests?
    
#### Smell tests: Is something fishy? (What you look for depends on the setting.)
    
    The only thing that would be fishy would be that there is no positive or negative sentiment for each measure there is only the measure itself. 
    
#### Do you have variation in the measures (i.e a variable is not all the same value)?
    
    Yes the variable is not all the same value.

#### Are the industries you expect talking about your subject positively or negatively?
    
    I did not groupby industry so I cannot tell what each industry is saying about my subject.

### Are there any caveats about the sample and/or data? If so, mention them and briefly discuss possible issues they raise with the analysis.

The main caveat with the data is that the return variable is not correct for multiple days. One issue this could raise is not being able to graph returns or understand it over a period of time. We only get a snapshot of the data at one moment in time instead of being able to see how it changes.

Another caveat with the data is that the "contextual sentiment" measures do not have positive and negative sentiments. I was unable to figure out how to correlate the positive words with my contextual variables in each 10-K so it is impossible to tell whether the 10-K's are talking about each "contextual sentiment" in a positive or negative way. One issue this could raise is looking at the way a 10-K is talking about one of the "contextual sentiments" and not knowing whether the 10-K is speaking positively or negatively about the measure. 

## Results

### Make a table with the correlation of each (10) sentiment measure against both (2) return measures. (So: an 10x2 table.)


```python
import pandas as pd
analysis = pd.read_csv('output/analysis_sample.csv')
table = analysis.drop(columns=['Symbol','CIK','Accession_number','Filing_date','10-K'])
table
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
      <th>LM_positive_sentiment</th>
      <th>LM_negative_sentiment</th>
      <th>ML_positive_sentiment</th>
      <th>ML_negative_sentiment</th>
      <th>Porter_five_sentiment</th>
      <th>Risk_var_sentiment</th>
      <th>Countries_sentiment</th>
      <th>ret</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003977</td>
      <td>0.023249</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>0.007929</td>
      <td>0.002211</td>
      <td>0.001256</td>
      <td>0.007573</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.003756</td>
      <td>0.012984</td>
      <td>0.024460</td>
      <td>0.023602</td>
      <td>0.009553</td>
      <td>0.002366</td>
      <td>0.002869</td>
      <td>-0.012737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003726</td>
      <td>0.012793</td>
      <td>0.021590</td>
      <td>0.024394</td>
      <td>0.009508</td>
      <td>0.002247</td>
      <td>0.001594</td>
      <td>-0.031431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.006481</td>
      <td>0.015448</td>
      <td>0.019753</td>
      <td>0.022645</td>
      <td>0.006563</td>
      <td>0.003428</td>
      <td>0.003346</td>
      <td>-0.006484</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.008642</td>
      <td>0.016861</td>
      <td>0.027968</td>
      <td>0.023964</td>
      <td>0.007103</td>
      <td>0.001906</td>
      <td>0.002040</td>
      <td>-0.007076</td>
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
    </tr>
    <tr>
      <th>486</th>
      <td>0.006078</td>
      <td>0.016549</td>
      <td>0.025014</td>
      <td>0.023482</td>
      <td>0.010269</td>
      <td>0.001828</td>
      <td>0.008203</td>
      <td>-0.009214</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.006258</td>
      <td>0.014964</td>
      <td>0.028396</td>
      <td>0.026842</td>
      <td>0.012282</td>
      <td>0.002256</td>
      <td>0.000894</td>
      <td>-0.077843</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.004591</td>
      <td>0.021783</td>
      <td>0.021506</td>
      <td>0.026759</td>
      <td>0.008550</td>
      <td>0.003173</td>
      <td>0.001818</td>
      <td>0.026077</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.003070</td>
      <td>0.013458</td>
      <td>0.016075</td>
      <td>0.016980</td>
      <td>0.003199</td>
      <td>0.001292</td>
      <td>0.002940</td>
      <td>-0.007552</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.005036</td>
      <td>0.019980</td>
      <td>0.021790</td>
      <td>0.033508</td>
      <td>0.008989</td>
      <td>0.003953</td>
      <td>0.002920</td>
      <td>0.006771</td>
    </tr>
  </tbody>
</table>
<p>491 rows × 8 columns</p>
</div>



### Include a scatterplot (or similar) of each sentiment measure against both return measures.




```python
import seaborn as sns

ax = sns.scatterplot(data=table,x='LM_positive_sentiment',y='ret')
ax.set(title='LM Positive Sentiment vs. Return',ylabel='Return',xlabel='LM Positive Sentiment')
```




    [Text(0.5, 1.0, 'LM Positive Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'LM Positive Sentiment')]




    
![png](output_14_1.png)
    



```python
ax = sns.scatterplot(data=table,x='LM_negative_sentiment',y='ret')
ax.set(title='LM Negative Sentiment vs. Return',ylabel='Return',xlabel='LM Negative Sentiment')
```




    [Text(0.5, 1.0, 'LM Negative Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'LM Negative Sentiment')]




    
![png](output_15_1.png)
    



```python
ax = sns.scatterplot(data=table,x='ML_positive_sentiment',y='ret')
ax.set(title='ML Positive Sentiment vs. Return',ylabel='Return',xlabel='ML Positive Sentiment')
```




    [Text(0.5, 1.0, 'ML Positive Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'ML Positive Sentiment')]




    
![png](output_16_1.png)
    



```python
ax = sns.scatterplot(data=table,x='ML_negative_sentiment',y='ret')
ax.set(title='ML Negative Sentiment vs. Return',ylabel='Return',xlabel='ML Negative Sentiment')
```




    [Text(0.5, 1.0, 'ML Negative Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'ML Negative Sentiment')]




    
![png](output_17_1.png)
    



```python
ax = sns.scatterplot(data=table,x='Porter_five_sentiment',y='ret')
ax.set(title='Porter Five Sentiment vs. Return',ylabel='Return',xlabel='Porter Five Sentiment')
```




    [Text(0.5, 1.0, 'Porter Five Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'Porter Five Sentiment')]




    
![png](output_18_1.png)
    



```python
ax = sns.scatterplot(data=table,x='Risk_var_sentiment',y='ret')
ax.set(title='Risk Sentiment vs. Return',ylabel='Return',xlabel='Risk Sentiment')
```




    [Text(0.5, 1.0, 'Risk Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'Risk Sentiment')]




    
![png](output_19_1.png)
    



```python
ax = sns.scatterplot(data=table,x='Countries_sentiment',y='ret')
ax.set(title='Countries Sentiment vs. Return',ylabel='Return',xlabel='Countries Sentiment')
```




    [Text(0.5, 1.0, 'Countries Sentiment vs. Return'),
     Text(0, 0.5, 'Return'),
     Text(0.5, 0, 'Countries Sentiment')]




    
![png](output_20_1.png)
    


### Four discussion topics:

On (1), (2), and (3) below: Focus just on the first return variable (which will examine returns around the 10-K publication)

On (4) below: Focus on how the “ML sentiment” variables (positive and negative) are related to the two different return measures.

(1) Compare / contrast the relationship between the returns variable and the two “LM Sentiment” variables (positive and negative) with the relationship between the returns variable and the two “ML Sentiment” variables (positive and negative). Focus on the patterns of the signs of the relationships and the magnitudes.

(2) If your comparison/contrast conflicts with Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo), discuss and brainstorm possible reasons why you think the results may differ. If your patterns agree, discuss why you think they bothered to include so many more firms and years and additional controls in their study? (It was more work than we did on this midterm, so why do it to get to the same point?)

(3) Discuss your 3 “contextual” sentiment measures. Do they have a relationship with returns that looks “different enough” from zero to investigate further? If so, make an economic argument for why sentiment in that context can be value relevant.

(4) Is there a difference in the sign and magnitude? Speculate on why or why not.

(1) The LM negative sentiment was more correlated than the LM positive sentiment while the ML positive and negative sentiments were very similar. They both correlated around 0 on returns while both ML's were around 2%. 

(2) I think mine differs because they were all concentrated around zero and were not correlated well with two different return variables. Because I only had the one variable that was not correct, the data was off in the graph.

(3) No they mostly have relationships around zero. 

(4) Not a ton. I think this is because the sentiment scores were all extremely low and concentrated near zero so there was not much difference between them. 
