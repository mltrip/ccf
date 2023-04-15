##  From https://www.reddit.com/r/binance/comments/k6b1r7/
##       accessing_earn_with_api/gu21o15/?utm_source=share
##       &utm_medium=web2x&context=3
import os
from pprint import pprint
import hmac
import hashlib
import requests
import json
import time

uri = "https://api.binance.com"

#  Fill in your Binance API key and Secret keys:
# binance_api_key = ""
# binance_api_secret = ""
binance_api_key = os.getenv('TRADE_API_KEY', None)
binance_api_secret = os.getenv('TRADE_SECRET_KEY', None)


def get_timestamp_offset():
    url = "{}/api/v3/time".format(uri)
    payload = {}
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, data=payload)
    result = json.loads(response.text)["serverTime"] - int(time.time() * 1000)
    return result

  
def generate_signature(query_string):
    m = hmac.new(binance_api_secret.encode("utf-8"), 
                 query_string.encode("utf-8"), 
                 hashlib.sha256)
    return m.hexdigest()

  
def get_system_status():
  timestamp = int(time.time() * 1000 + get_timestamp_offset())
  query_string = f"timestamp={timestamp}"
  signature = generate_signature(query_string)
  url = f"{uri}/sapi/v1/system/status?{query_string}&signature={signature}"
  payload = {}
  headers = {"Content-Type": "application/json",
             "X-MBX-APIKEY": binance_api_key}
  result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
  return result  
  
  
def get_account_status():
  timestamp = int(time.time() * 1000 + get_timestamp_offset())
  query_string = f"timestamp={timestamp}"
  signature = generate_signature(query_string)
  url = f"{uri}/sapi/v1/account/status?{query_string}&signature={signature}"
  payload = {}
  headers = {"Content-Type": "application/json",
             "X-MBX-APIKEY": binance_api_key}
  result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
  return result
  
  
def get_api_trading_status():
  timestamp = int(time.time() * 1000 + get_timestamp_offset())
  query_string = f"timestamp={timestamp}"
  signature = generate_signature(query_string)
  url = f"{uri}/sapi/v1/account/apiTradingStatus?{query_string}&signature={signature}"
  payload = {}
  headers = {"Content-Type": "application/json",
             "X-MBX-APIKEY": binance_api_key}
  result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
  return result
  
  
def get_flexible_savings_balance(asset):
    """ Get your balance in Bincance Earn :: Flexible Savings """
    timestamp = int(time.time() * 1000 + get_timestamp_offset())
    query_string = "asset={}&timestamp={}".format(asset, timestamp)
    signature = generate_signature(query_string)
    url = "{}/sapi/v1/lending/daily/token/position?{}&signature={}".format(
           uri, query_string, signature)
    payload = {}
    headers = {"Content-Type": "application/json",
               "X-MBX-APIKEY": binance_api_key}
    result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
    return result

  
def get_locked_savings_balance(asset, project_id):
    """ Get your balance in Bincance Earn :: Locked Savings """
    timestamp = int(time.time() * 1000 + get_timestamp_offset())
    query_string = "asset={}&projectId={}&status=HOLDING&timestamp={}".format(
                    asset, project_id, timestamp)
    signature = generate_signature(query_string)

    url = "{}/sapi/v1/lending/project/position/list?{}&signature={}".format(
              uri, query_string, signature)
    payload = {}
    headers = {"Content-Type": "application/json",
               "X-MBX-APIKEY": binance_api_key}
    result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
    return result

  
def get_all_earn_products():
    """ Gets all savings products from Binance """
    def get_earn_products(current_page=1):
        """ Gets 50 savings products in "current" page ...modified from source:
            https://binance-docs.github.io/apidocs/spot/en/#savings-endpoints """
        timestamp = int(time.time() * 1000 + get_timestamp_offset())
        query_string = "&current={}&status=SUBSCRIBABLE&timestamp={}".format(
                        current_page, timestamp)
        signature = generate_signature(query_string)

        url = "{}/sapi/v1/lending/daily/product/list?{}&signature={}".format(
                  uri, query_string, signature)
        payload = {}
        headers = {"Content-Type": "application/json",
                  "X-MBX-APIKEY": binance_api_key}
        result = json.loads(requests.request("GET", url, headers=headers, data=payload).text)
        return result
    all_products = []
    more_products = True
    current_page = 0
    while more_products:
        current_page += 1
        prod = get_earn_products(current_page=current_page)
        all_products.extend(prod)
        if len(prod)==50:
            more_products = True
        else:
            more_products = False
    return all_products

  
if __name__=="__main__":
  status = get_system_status()
  pprint(status)
  status = get_account_status()
  pprint(status)
  status = get_api_trading_status()
  pprint(status)
  
  # flex = get_flexible_savings_balance("BTC")
  # print (flex)
  # print ('')

  # lock = get_locked_savings_balance("FUN", "CFUN14DAYSS001")
  # print (json.dumps(lock, indent=2))
  # print ('')

  # earn_products = get_all_earn_products()
  # print ("There are", len(earn_products), "subscribable earn products.\n")
  # print ("The frist two of them are:\n\n", 
  #        json.dumps(earn_products[0:2], indent=2))