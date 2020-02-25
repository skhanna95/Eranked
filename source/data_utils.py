import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
from selenium import webdriver
import time
import re
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from tqdm import tqdm

def makeTrainingData(train_csv,decription_csv,attrib_csv,merge_col,encoding='ISO-8859-1'):
    """
    Making training data by combing train.csv,
    product product_descriptions.csv and attributes.csv
    """
    
    train_df = pd.read_csv(os.path.join('data','homedepot',train_csv), encoding=encoding)
    description_df = pd.read_csv(os.path.join('data','homedepot',decription_csv), encoding=encoding)
    attrib_df = pd.read_csv(os.path.join('data','homedepot',attrib_csv), encoding=encoding)
    
    brand_df = attrib_df[attrib_df.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

    merged = train_df.merge(brand_df,how='left',on=merge_col)
    return merged.merge(description_df,how='left',on=merge_col)

def rankByRelevance(pd_df,search_term='search_term',relevance_col='relevance'):
    """
    Sort the products by search term and relevance
    and rank withing each group based on relevance.
    """
    pd_df = pd_df.sort_values([search_term,relevance_col],ascending=False)
    pd_df['rank'] = pd_df.groupby(search_term).cumcount()+1
    
    return pd_df

def cleanData(value,stem_lib='porterstemmer',lower=True):
    """
    Call to remove special chars and atrributes data,
    specify if we need lower and stem the words
    based on the stemmer.
    """
    if isinstance(value, float):
        return ['<UNK>']

    if '<unk>' == value.lower():
        return ['<UNK>']
    if lower:
        value = [word for word in removeAdditional(value).lower().split()]
    else:
        value = [word for word in removeAdditional(value).split()]
    
    stem_lib = stem_lib.lower()

    if stem_lib == 'none':
        return value
    
    stemmer = None
    if stem_lib == 'porterstemmer':
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in value]
        
    if stem_lib == 'lancasterstemmer':
        stemmer = LancasterStemmer()
        return [stemmer.stem(word) for word in value]

def cleanPandasDF(ranked_df,col_list_clean,col_list_drop=[]):
    
    """
    Applying the clean data function to all the column
    in the pandas dataframe, and droppin irrelevant columns
    """
    
    for cl in tqdm(col_list_clean,desc="cleanPandasDF"):
        ranked_df[cl] = ranked_df[cl].apply(lambda x: tuple(cleanData(x)))
    
    if len(col_list_drop):
        ranked_df = ranked_df.drop(col_list_drop, axis = 1, inplace = False).reset_index().drop(['index'],axis=1)
    
    return ranked_df

def fetchBrandName(url,source_name,unk='<UNK>'):
    """
    Fetch different brand names from various sources
    using web scrapping.
    """
    if source_name == 'Overstock':
            driver = webdriver.Chrome('./chromedriver')
            driver.get(url)
            time.sleep(1)

            os_soup = BeautifulSoup(driver.page_source)
            driver.quit()
            if os_soup:
                span_tag = os_soup.find('span', attrs={'id': 'brand-name'})
                if span_tag:
                    a_span = span_tag.find('a')
                    if a_span:
                        return a_span.text
            return unk
    
    
    response = requests.get(url)
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if source_name == 'eBay':
            if soup:
                soup_child = soup.find('h2', itemprop = 'brand')

                if soup_child:
                    soup_text = soup_child.find('span', itemprop = 'name')                
                    if soup_text:
                        return soup_text.text

                second_find = soup.find_all('li', class_ = '')
                if second_find:
                    for sf in second_find:
                        if sf.find('div', class_ = 's-name'):
                            if sf.find('div', class_ = 's-name').text.lower() == 'brand':
                                if sf.find('div', class_ = 's-value'):
                                    return sf.find('div', class_ = 's-value').text
        elif source_name == 'wallmart':
            w_brand = soup.find('span',itemprop='brand')
            if w_brand:
                return w_brand.text
            
        elif source_name == 'Target':
            options = webdriver.ChromeOptions()
            options.add_argument("headless")

            driver = webdriver.Chrome('./chromedriver',options=options)
            driver.get(url)
            time.sleep(3)
            
            t_soup = BeautifulSoup(driver.page_source)
            a_child = t_soup.find('a', attrs={'data-test': 'shopAllBrandLink'})
            if a_child:
                span_div = a_child.find('span')
                if span_div:
                    return span_div.text.split(' all ')[1]
                

    return unk

#https://gist.github.com/susanli2016/b83d148de7394821509bd5172d2c96d3#file-str_stem
def removeAdditional(value): 
    if isinstance(value, str):
        value = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", value)
        value = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", value)
        value = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", value)
        value = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", value)
        value = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", value)
        value = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", value)
        value = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", value)
        value = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", value)
        value = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", value)
        value = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", value)
        value = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. ", value)
        value = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. ", value)
        value = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. ", value)
        value = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", value)
        value = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", value)
        value = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr ", value)
        value = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. ", value)
        value = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ", value)
        
        value = value.replace("$"," ")
        value = value.replace("?"," ")
        value = value.replace("&nbsp;"," ")
        value = value.replace("&amp;","&")
        value = value.replace("&#39;","'")
        value = value.replace("/>/Agt/>","")
        value = value.replace("</a<gt/","")
        value = value.replace("gt/>","")
        value = value.replace("/>","")
        value = value.replace("<br","")
        value = value.replace("<.+?>","")
        value = value.replace(r"[ &<>)(_,;:!?\+^~@#\$]+"," ")
        value = value.replace("'s\\b","")
        value = value.replace("[']+","")
        value = value.replace("[\"]+","")
        value = value.replace("-"," ")
        value = value.replace("+"," ")
        
        value = value.replace("[ ]?[[(].+?[])]","")
        
        value = value.replace("size: .+$","")
        value = value.replace("size [0-9]+[.]?[0-9]+\\b","")
        value = re.sub('[^A-Za-z0-9-./]', ' ', value)
        value = value.replace('  ',' ')
        
        return value
    else:
        return ""