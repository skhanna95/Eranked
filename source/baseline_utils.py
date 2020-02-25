import requests
import re
from random import randint
import time
from spellchecker import SpellChecker
from os import path
import json

class SpellingCorrectionBaseline:
    def __init__(self,filename='query_corr.csv',sep='||',header=True,header_del_col='query'):
        self.path = 'data/' + filename
        self.header=header
        self.sep = sep
        self.header_del_col = header_del_col
        self.spelling_ref_dict = self.checkQueryDict()
        self.append_values = {}
    
    def dicAppender(self):
        if len(self.append_values) > 0:
            with open(self.path, "a") as myfile:
                for k,v in self.append_values.items():
                    myfile.write('\n'+k+self.sep+v)
            
    def checkQueryDict(self):
        if path.exists(self.path):
            spelling_ref_dict = {}
            with open(self.path,'r') as lines:
                for line in lines:
                    split_line = line.split(self.sep)
                    spelling_ref_dict[split_line[0]]=split_line[1].replace('\n','')
            if self.header:
                del spelling_ref_dict[self.header_del_col]
            return spelling_ref_dict
        
        return {}
    
    def spellCorrectBaseline(self,check_str):
        """
        Baseline spell checker
        uses google
        """
        checker = self.spelling_ref_dict.get(check_str)
        if checker:
            return checker

        r = requests.get("https://www.google.com/search?q="+ check_str.replace(' ','+'))

        try:
            r_lower = r.text.lower()
            d_mean = 'did you mean:'
            sh_res = 'showing results for'

            if d_mean in r_lower:
                s_ix = r_lower.find(d_mean) 
                e_ix = r_lower.find('</a></div></div>')
                subset_text = r.text[s_ix+len(sh_res)+1:e_ix]
                s_s_ix = subset_text.find('>')
                self.append_values[check_str] = re.sub(r'<[^>]+>', '', subset_text[s_s_ix+1:])
                return re.sub(r'<[^>]+>', '', subset_text[s_s_ix+1:])

            if sh_res in r_lower:
                s_ix = r_lower.find(sh_res) 
                e_ix = r_lower.find('</a><script nonce=')
                subset_text = r.text[s_ix+len(sh_res)+1:e_ix]
                s_s_ix = subset_text.find('>')
                self.append_values[check_str] = re.sub(r'<[^>]+>', '', subset_text[s_s_ix+1:])
                return re.sub(r'<[^>]+>', '', subset_text[s_s_ix+1:])
            
            if 'detected unusual traffic from your computer network' in r_lower:
                print(check_str)
                return self.spellCorrectBackupBaseline(check_str)
            
            self.append_values[check_str] = check_str
            return check_str
        
        except Exception as e:
            print('Exception:',e)
            w_t = randint(0,2)
            print('Retry:',check_str,'in',w_t,'seconds')
            time.sleep(w_t)
            return self.spellCorrectBaseline(check_str)
        
    def spellCorrectBackupBaseline(self,check_str):
        """
        Baseline spell checker
        uses spellchecker library
        """
        print('spellCorrectBackupBaseline called')
        spell = SpellChecker()
        spell.known(['zwave', 'rheem']) 
        splitted = check_str.split()

        for w_ix in range(len(splitted)):
            if splitted[w_ix].isalpha():
                mis_check = list(spell.unknown([splitted[w_ix].lower()]))
                if len(mis_check) == 1:
                    splitted[w_ix] = spell.correction(mis_check[0])
                    
        final_result = " ".join(splitted)
        # self.append_values[check_str] = final_result
        
        return final_result

    def spellingCheckNew(self,query):
        
        check_str = query
        checker = self.spelling_ref_dict.get(check_str)
        if checker:
            return checker

        api_key = "693bb6e8c10f47f39f89973910d135c2"
        endpoint = "https://rms2.cognitiveservices.azure.com/bing/v7.0/spellcheck"
        
        data = {'text': query}
        
        params = { 'mkt':'en-us', 'mode':'spell'}
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Ocp-Apim-Subscription-Key': api_key,
        }
        
        response = requests.post(endpoint, headers=headers, params=params, data=data)
        
        json_response = response.json()
        
        for tkns in json_response['flaggedTokens']:
            query = query.replace(tkns['token'],tkns['suggestions'][0]["suggestion"])

        self.append_values[check_str] = query
        
        return query