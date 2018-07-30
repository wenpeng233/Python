#!/usr/bin/env python
import requests
from bs4 import BeautifulSoup
import re


url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq" \
      "=1531555796362_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=美女"
response = requests.get(url)
# print(response.text)
soup = BeautifulSoup(response.text, 'html.parser')
div = soup.find_all("div",id="wrapper")[0].find_all("div", id="imgContainer")[0]
# for iframe in iframexx:
#     responsexx = requests.get(iframe.attrs['src'])
#     soup = BeautifulSoup(responsexx.text, "html.parser")
print(div)
# print(soup.prettify())
# print(soup.find_all("#document"))
# for child in soup.body.descendants:
#     print(child)
div_links = soup.find_all("body")


# print(div_links)