import os
import html2text
import json
import re
import pandas as pd
import magic
import codecs
import xml.etree.ElementTree as ET
from boilerpy3 import extractors


def getContentType(filePath):
    if not os.path.exists(filePath):
        return 'unknown'
    contentType = magic.from_file(filePath, mime=True).lower()

    with codecs.open(filePath, "r", "utf_8_sig" ) as f:        
        if 'empty' in contentType:
            return 'empty'
        elif 'xml' in contentType:
            return 'xml'
        elif 'html' in contentType:
            return 'html'
        elif 'json' in contentType or f.read(1) in '{[':
            return 'json'
        elif 'text' in contentType:
            return 'txt'
        else:
            return 'unknown'


def getContentForDf(rawDataPath, categoryName):    
    filenameArr = []
    categoryArr = []
    contentArr = []

    with os.scandir(rawDataPath) as entries:
        for entry in entries:
            f = codecs.open(entry, "r", "utf_8_sig")
            data = f.read()

            # print(entry.name)
            if getContentType(rawDataPath + entry.name) == 'xml':
                arr = []
                xmlstr = data.replace('&', '')
                root = ET.fromstring(xmlstr)                
                for item in list(root):
                    arr.append(item.text.encode('utf8'))
              
                lenArr = [len(str(el)) for el in arr]
                content = arr[lenArr.index(max(lenArr))]
            
            elif getContentType(rawDataPath + entry.name) == 'txt':
                content = data

            elif getContentType(rawDataPath + entry.name) == 'json':
                arr = []
                data = data.replace('@', '')
                data = re.sub(r"(?m)^\s+", "", data)

                entryContent = json.loads(data)
                if type(entryContent) is list:
                    arr.extend(entryContent)
                else:
                    for key, value in entryContent.items():
                        arr.append(value)
                
                lenArr = [len(str(el)) for el in arr]
                content = arr[lenArr.index(max(lenArr))]

            elif getContentType(rawDataPath + entry.name) == 'html':
                # content = html2text.html2text(data)
                extractor = extractors.ArticleExtractor()
                content = extractor.get_content(data)

            else:
                content = ''
                
            filenameArr.append(entry.name)
            categoryArr.append(categoryName)
            contentArr.append(content)

    return filenameArr, categoryArr, contentArr


infoRawDataPath = os.getcwd() + '\\rawdata\\info\\'
supportRawDataPath = os.getcwd() + '\\rawdata\\support\\'
unknownRawDataPath = os.getcwd() + '\\rawdata\\unknown\\'

filenameArr, categoryArr, contentArr = getContentForDf(infoRawDataPath, 'info')
filenameArr2, categoryArr2, contentArr2 = getContentForDf(supportRawDataPath, 'support')
filenameArr3, categoryArr3, contentArr3 = getContentForDf(unknownRawDataPath, 'unknown')

filenameArr.extend(filenameArr2)
categoryArr.extend(categoryArr2)
contentArr.extend(contentArr2)

filenameArr.extend(filenameArr3)
categoryArr.extend(categoryArr3)
contentArr.extend(contentArr3)

df = pd.DataFrame({'File name': filenameArr,
                    'Content': contentArr,
                    'Category': categoryArr})
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(os.getcwd() + '\\data.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1', index=False)
# Close the Pandas Excel writer and output the Excel file.
writer.save()
