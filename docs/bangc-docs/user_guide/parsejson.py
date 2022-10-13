#!/usr/bin/python
# -*- coding: UTF-8

import os
import sys
import json
import re
import codecs
import shutil
import platform
import importlib

Py_version = sys.version_info
Py_v_info = str(Py_version.major) + '.' + str(Py_version.minor) + '.' + str(Py_version.micro)

if Py_version >= (3,5,0) and Py_version <(3,7,0):
    print(False)
    
if Py_version >= (3,7,0):
    print(True)
    
if platform.system().lower() == 'windows':
    print("windows")
elif platform.system().lower() == 'linux':
    print("linux")

filepathlist=os.path.split(os.path.realpath(__file__))
currfilepath=filepathlist[0]

def __openconfjsonfile__():
    jsonfile = currfilepath+'/conf.json'
    with codecs.open(jsonfile,"r+",encoding = 'utf-8') as load_f:
        load_dict = json.load(load_f)
        return load_dict

def getignorewarning():
    return __load_dict__["ignorewarndes"],__load_dict__["ignorewarnkey"]
       
def GetOtherIncludePackage():
#    packagearr = __load_dict__['package']
#    for package in packagearr:
#        print(package)
    return __load_dict__['package']

def GetReplacePackage():
    return __load_dict__['replacepackage']

def GetCustomOptions():
    return __load_dict__['customoptions']

def GetIsTocContents():
    return __load_dict__['isfiguretabletoc']

def GetSensitiveword():
    #得到敏感词数组，便于搜索文档中是否有敏感词存在
    return __load_dict__['sensitivewords']

def GetTablesContent():
    return __load_dict__['tables']

def GetTablerowtype():
#    packagearr = __load_dict__['tables']['rowtype']
#    print(packagearr)
    return __load_dict__['tables']['rowtype']

def GetTableheadtype():
#    packagearr = __load_dict__['tables']['headtype']
#    print(packagearr)
    return __load_dict__['tables']['headtype']

def GetTableHeadFontColor():
    return __load_dict__['tables']['headfontcolor']

def GetTableStylesArr():
#    packagearr = __load_dict__['tables']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['tables']['styles']

def GetImageStyleArr():
#    packagearr = __load_dict__['image']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['image']['styles']

#判断是否有忽略告警的类
class clsIgnoreWarn:
    
    def __init__(self,warnfile):
        self.warnfile = warnfile  #保存告警文件

    #判断该关键字是否在告警中，在告警中则返回true，不在告警中则返回fasle
    def __JudgeWarnKey(self,keystr,warnstr):
        
        #先判断是否为组合关键字
        if "&&" in keystr:
            #解析组合关键字为字符串列表
            keyls = keystr.split("&&")
            isignore = True  #默认该告警为忽略，如果组合关键字其中一个关键字没匹配上，则该告警不能忽略。
            for i in range(0,len(keyls)):
                if keyls[i].strip().lower() not in warnstr.lower():
                    isignore =False
                    break
            return isignore
        else:
            if keystr.lower() in warnstr.lower(): 
                return True  #忽略告警在告警字符串里面，则返回true，表示该告警可以忽略
            else:
                return False
    
    #解析告警文件
    def parsewarnfile(self,ignorelist,ignorekey):
        '''
        #make latex命令产生的告警都会保存在stderr文件中，只要判断该文件中的告警是否在忽略告警里就可以了，不在忽略告警里，则肯定有错误，停止执行
        '''
        if not os.path.exists(self.warnfile):
            return True
        
        fs = codecs.open(self.warnfile,"r",encoding = 'utf-8')
        fstr = fs.read()   #先将所有内容读取到字符串中，方便正则表达式查找
        fs.close
    
        #查找带warning的内容
        pattern = re.compile(r"([\s\S].*)WARNING:([\s\S].*)", re.I | re.U)
        mobjarr = pattern.finditer(fstr)
        
        errstr = '' #保存不能被忽略的告警
        isNotError = True
        for mobj in mobjarr:
            amarr = mobj.group()
            
            if amarr == '':
                    continue
                
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            
            #print("pdfparse: \n%s" % amarr)
            #判断该告警是否在忽略列表里，不在忽略列表里，保存在errstr字符串中
            if amarr.lower() not in [elem.lower() for elem in ignorelist]: #如果错误不在忽略告警里面
                #如果不能整行匹配，再判断该行是否包含需忽略的告警的关键字
                isWrite = False  #默认不能忽略
                for igkey in ignorekey:
    
                    isWrite = self.__JudgeWarnKey(igkey,amarr) 
                    if isWrite:
                        break;
                    
                if isWrite == False:
                    #写入stderr文件
                    isNotError = False
                    #print("make latex Error description: \n%s" % amarr)
                    #fe.writelines(amarr+'\n')
                    errstr += amarr
                    
        if errstr != '':
            #如果有不能被忽略的告警，则将不能忽略的告警重新写入warnfile文件中
            #先删除源文件，避免原文件无法写入
            fw = codecs.open(self.warnfile, "w",encoding = 'utf-8')
            fw.write(errstr)
            fw.close
        else:
            #如果所有的告警都能忽略则删除之前的告警文件
            #该功能暂时不实现，还是保存原始的告警文件，方便查找。
            #os.remove(self.warnfile)
            pass
           
        return isNotError
    
    #该函数暂时未用
    def __parsepdfarg(stdout,stderr,ignorelist,ignorekey):
        '''
        make all-pdf命令产生的所有输出都只会保存在stdout中，因此从stdout中判断是否有告警内容
        '''
        stdoutfilepath = currfilepath+'/'+stdout
        stderrfilepath = currfilepath+'/'+stderr
    
        if not os.path.exists(stdoutfilepath):
            return True
        
        fs = codecs.open(stdoutfilepath, "r+",encoding = 'utf-8')
        fstr = fs.read()   #先将所有内容读取到字符串中，方便正则表达式查找
        fs.close
        #查找latexmk的位置，latexmk位置的开始即make all-pdf打印输出的开始
        searchstr = r"latexmk"
        m = re.search(searchstr, fstr, re.I|re.U )
        if m == None:
            return True
        
        spos = m.span()  #获取位置
        latexcont = fstr[spos[0]:len(fstr)]  #获取到make all-pdf产生的内容
        
        #查找带warning的内容
        pattern = re.compile(r'([\s\S].*)Warning([\s\S].*)',  re.I|re.U)
        mobjarr = pattern.finditer(latexcont)
        
        #打开stderr文件，方便后续写
        fe = codecs.open(stderrfilepath, "a+",encoding = 'utf-8')
        isNotError = True
        for mobj in mobjarr:
            amarr = mobj.group()
            #print("pdfparse: %s \n" % amarr)
            if amarr == '':
                    continue
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            
            #判断该告警是否在忽略列表里，不在忽略列表里，要写入stderr文件
            if amarr.lower() not in [elem.lower() for elem in ignorelist]: #如果错误不在忽略告警里面
                #如果不能整行匹配，再判断该行是否包含需忽略的告警的关键字
                isWrite = False  #默认不能忽略
                for igkey in ignorekey:
    
                    isWrite = __JudgeWarnKey(igkey,amarr) 
                    if isWrite:
                        break;
                    
                if isWrite == False:
                    #写入stderr文件
                    isNotError = False
                    #print("make all-pdf Error description: \n%s" % amarr)
                    fe.writelines(amarr)
        fe.close 
           
        return isNotError

def warn_main(warnfile):
    
    ignorelist,ignorekey = getignorewarning()
    clswarn = clsIgnoreWarn(warnfile)
    
    if not clswarn.parsewarnfile(ignorelist,ignorekey):
        #如果存在不可忽略的告警，则返回True，否则返回False
        return True
        
    return False
    

class clsTableattr:

    def __init__(self, tables):
        self.rowtype = tables['rowtype']
        self.headtype = tables['headtype']
        self.headfontcolor = tables['headfontcolor']
        self.tablestyles = tables['styles']
        self.isname = tables['isname']

class clsModifyTex:

    def __init__(self, content):
        self.content = content
        self.tablesattrobj = clsTableattr(GetTablesContent())

    #加入其它包
    def AddPackageToTex(self):
        #得到需要包的数组
        packarr = GetOtherIncludePackage()
        if len(packarr)==0:
            return  False;

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        #searchstr = r'\\usepackage\[dontkeepoldnames\]{sphinx}'
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)
        
        replacestr=""
        for package in packarr:
            replacestr  += package+'\n'
             
        if Py_version >= (3,7,0):
            replacestr += "\\" + matchstr.group(0)
        else:
            replacestr += matchstr.group(0)
        
       
        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)

        return True

    #加入自定义选项,包放在了sphinx包的前面，因此选项放在sphinx包的后面
    def AddCustormOptionsToTex(self):
        #得到需要包的数组
        packarr = GetCustomOptions()
        if len(packarr)==0:
            return  False;

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将自定义参数放在它的后面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)

        replacestr=""
        for package in packarr:
             replacestr  += package+'\n'
             
        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0)+'\n'+replacestr
        else:
            replacestr = matchstr.group(0)+'\n'+replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True
     
    #增加figure和table toc到tex
    def AddOtherTocToTex(self):
        #得到需要包的数组
        packarr = GetIsTocContents()
        if len(packarr)==0:
            return  False;

        replacestr = ""
        if packarr['isfigurestoc']:
           figlst = packarr['figurestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'

        if packarr['istablestoc']:
           figlst = packarr['tablestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'
        if replacestr == "":
           return

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\sphinxtableofcontents'
        matchstr = re.search(searchstr,self.content)

        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0) + '\n' + replacestr
        else:
            replacestr = matchstr.group(0) + '\n' + replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True


    #得到需要替换的包，用正则表达式替换
    def ModifyReplacePackage(self):
        #得到字典值
        redict = GetReplacePackage()
        if len(redict) ==0:
           return;
        
        #返回字典中所有键值的列表
        keylst = list(redict)
        for key in keylst:
            if key == 'comment' :
                continue;
            keyvalue = redict[key]   #得到键对应的值
            
            #对键值进行替换
            self.content = re.sub(key, keyvalue, self.content, 0, re.M | re.I|re.U)
        return;
    
    def __ModifyFunctionBkColorByPython__(self,newcontent,linecontent,curpos,prepos):
        #根据python生成的格式为类和函数声明添加灰底，并按逗号换行
        
        '''
        根据正则表达式获得以\pysiglinewithargsret开头，以“{}”结尾的中间字符串
        对中间字符串添加灰底和换行
        '''
        
        startmultiline = '\n\\begin{shaded}\n\\pysigstartmultiline\n'
        stopmultiline = '\n\\pysigstopmultiline\n\\end{shaded}'
        #searchstr = r'(?=\\pysiglinewithargsret).*(?<={})'
        searchstr=r'(?=\\pysiglinewithargsret).*'
        match = re.search(searchstr, linecontent, re.I|re.U)
        if match != None:
           #print('modstr = %s' % match.group())
           #重新组合linecontent
           pos = match.span()
           newstr = '\n' + linecontent[:match.start()] + startmultiline + match.group().replace(r",",r",\\") + stopmultiline + linecontent[match.end():len(linecontent)]
           #计算替换前的内容
           if len(prepos)==0:
               newcontent = self.content[:curpos[0]-1] + newstr
           else:
               newcontent += self.content[prepos[1]:curpos[0]-1] + newstr
           
        return newcontent
    
    #从指定字符串开始对行内字符串进行替换
    '''
    srcstr：原始字符串
    posstr：从该字符串开始进行替换
    oldstr：被替换字符串
    newstr：替换字符串
    '''
    def __strreplacepos(self,srcstr,posstr,oldstr,newstr):
        
        #查找字符串的其实位置
        pos = srcstr.find(posstr)
        if pos == -1:
            return ""  #如果查找的字符串没有找到，则返回空。
        
        #根据找到的位置进行字符串拆分
        startstr = srcstr[0:pos]
        beforestr = srcstr[pos:len(srcstr)]
        #对字符串进行替换
        afterstr = beforestr.replace(oldstr,newstr)
        return startstr+afterstr 
        
    #该函数用来修改函数的背景色，并根据参数换行
    def ModifyFunctionBackColor(self):
        '''
        * c/c++ sphinx生成的函数都被以下三行包围：
          \pysigstartmultiline
          \pysiglinewithargsret{xxxxx}
          \pysigstopmultiline
          因此利用正则表达式查找这三行所在的位置进行替换。
          注意：
          要实现该功能，latex必须包含framed和color包，同时定义以下颜色：
          \definecolor{shadecolor}{RGB}{220,220,220}
          如果shadecolor颜色未定义，则该函数执行失败。
        * python生成的latex文件中函数或者类的声明，有\pysiglinewithargsret指令，但是没有pysigstartmultiline指令。
          因此需要在\pysiglinewithargsret指令句子的结束，前后先添加pysigstartmultiline和pysigstopmultiline。再添加\begin{shaded}和\end{shaded}
          一句的结束根据"{}"来判断，遇到"{}"即为句末。
        '''
        pythontype = False
        newcontent = ""
        prepos=[]     #定义需要替换的字符串列表，该列表中的每一个元素都包含上面提到的三行。
        searchstr = r'^(?=.*\\pysiglinewithargsret).*$'
        m = re.finditer(searchstr, self.content, re.M|re.I|re.U)
        for match in m:
            
            linecontent = match.group()
            #判断是否添加了pysigstartmultiline和pysigstopmultiline，没有添加的话需要添加才能添加灰色背景
            multilen=len(r'\pysigstartmultiline')
            startmultistr = self.content[match.start()-multilen-1:match.start()]
            #print('startmultistr : %s' % startmultistr)
            #如果上一行不是\pysigstartmultiline则需要按python风格修改，添加该标志
            if startmultistr.strip() != r'\pysigstartmultiline':
                #print('is python')
                newcontent = self.__ModifyFunctionBkColorByPython__(newcontent,linecontent,match.span(),prepos)
                prepos = match.span()
                pythontype = True
            else:
                #tablestr = match.groups() 
                #计算替换前的内容
                if len(prepos)==0:
                    newcontent = self.content[0:match.start()-len(r'\pysigstartmultiline')-1]
                else:
                    newcontent += self.content[prepos[1]+len(r'\pysigstopmultiline')+1:match.start()-len(r'\pysigstartmultiline')-1]
                
                #当有template的时候，pysiglinewithargsret不一定在行首，因此需要从pysiglinewithargsret开始替换逗号，加强制换行符
                afterstr=self.__strreplacepos(linecontent,r"pysiglinewithargsret",r",",r",\\")
                #得到替换后的字符串
                newstr = '\\begin{shaded}\n' + '\\pysigstartmultiline\n' + afterstr +'\n\\pysigstopmultiline\n'+'\\end{shaded}'
                #得到替换后的内容
                newcontent += newstr
                prepos = match.span() 
                pythontype = False

        if len(prepos) > 0:
            if not pythontype:
                self.content = newcontent + self.content[prepos[1]+len(r'\pysigstopmultiline')+1:len(self.content)]
            else:
                self.content = newcontent + self.content[prepos[1]+1:len(self.content)]
            
    def ModifyTablesAttributes(self):
        #修改表格属性
        newcontent = self.content
        searchstr = r'(\\begin{savenotes}\\sphinxattablestart|\\begin{savenotes}\\sphinxatlongtablestart)([\s\S]*?)(\\sphinxattableend\\end{savenotes}|\\sphinxatlongtableend\\end{savenotes})'
        m = re.finditer(searchstr, self.content, re.M|re.I|re.U)
        for match in m:
            oldtablestr = match.group()
            tablestr = match.groups()
            caption_dict = self.__CreateTableCaptionDict(self.tablesattrobj.tablestyles)
            if len(caption_dict ) > 0 :
                newtableattr = self.__ModifySingleTableattr(tablestr[0]+tablestr[1]+tablestr[2],caption_dict ) #tablestr也是3个内容的数组，因为正则表达式被分为了3组，只取中间分组的内容。
                #重新组成新的字符串
                newcontent = newcontent.replace(tablestr[0]+tablestr[1]+tablestr[2], newtableattr)
                
        self.content = newcontent


    def __CreateTableCaptionDict(self, tablestylesarr):
        #根据caption生成表格字典，key=caption，value=属性数组
        cap_dict = {}

        for tablestyle_dict in tablestylesarr:
            captionarr = tablestyle_dict['caption']
            #该caption可能是一个逗号分隔的字符串数组，因此需要做拆分
            captionlist = captionarr.split(",")
            for caption in captionlist:
                cap_dict[caption] = tablestyle_dict  #以caption为key重新生成字典，便于查找
        return cap_dict

    def __ModifySingleTableattr(self, singletablecontent, caption_dict):
        #修改单个表格属性
        #从单个表格里用正则表达式找caption
        #定义正则表达式,查找caption内容
        new_singletablecontent = singletablecontent
        if self.tablesattrobj.isname:
            searchstr = r'.*\\label.*?:(?P<caption>[\s\S].*)}}.*'
        else:
            searchstr = r'(\\sphinxcaption|\\caption){(?P<caption>[\s\S]*?)}'
        
        matchcaption = re.search(searchstr, singletablecontent, re.M | re.I|re.U)
        if matchcaption != None:
            tablecaption = matchcaption.group('caption') #得到caption的值
        else:
            tablecaption = ''
        if tablecaption in caption_dict:
            #修改表格属性
            tablestyle_dict = caption_dict[tablecaption]

                #修改表格为长表格
            new_singletablecontent = self.__StartModifyTableAttr(singletablecontent, 
                                                                 tablestyle_dict['isLongTable'],
                                                                 tablestyle_dict['isCusHead'])
            #渲染竖型表格的第一列
            if tablestyle_dict['isVertical']==True:
                new_singletablecontent = self.__ModifyVerticalTable(new_singletablecontent)
        else:
            #修改表格的通用属性
            new_singletablecontent = self.__StartModifyTableAttr(singletablecontent, False)
        if new_singletablecontent == '':
           new_singletablecontent = singletablecontent

        return new_singletablecontent

    def __StartModifyTableAttr(self, singletablecontent, islongtable,isCusHead=True):
        #修改表格属性
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        #为了添加表格的通用属性，先对字符串做分割
        #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
        #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},第2维字符串直接删除，第3维字符串不变
        splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
        if splittable == None or len(splittable) < 5:
           	#再修改长表格属性
            searchstr = r'\\begin{longtable}([\s\S].*)'
            #为了添加表格的通用属性，先对字符串做分割
            #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
            #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},第2维字符串直接删除，第3维字符串不变
            splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
            if len(splittable) < 3 or isCusHead==False:
                #至少是3维的数组，否则不是预想的内容，不做处理
                return singletablecontent
            newtable4 = self.__ModifyLongTableHead(splittable[2], self.tablesattrobj.headtype)
            singletablecontent = splittable[0]+r'\begin{longtable}'+splittable[1]+newtable4  #begin{longtable}必须再加上，因为Python并不认为它是正则表达式，因此不再分组里面第0个分组为空。
            return singletablecontent

        #拆分后splittable应该为5个字符串的数组，拆分后便于添加通用属性
        if self.tablesattrobj.rowtype != '':
            splittable[0] += self.tablesattrobj.rowtype + '\n'

        if isCusHead == True:
            #修改表头字体颜色为白色加粗
            newtable4 = self.__ModifyTableHead(splittable[4], self.tablesattrobj.headtype)
        else:
            newtable4 = splittable[4]
        
        singletablecontent = splittable[0]+splittable[1]+splittable[2]+splittable[3]+newtable4
        if islongtable: #如果为长表格要做长表格的替换
            singletablecontent = self.__ModifyTableLongHeadAndTail(singletablecontent)

        return singletablecontent
    
    def __ModifyTableLongHeadAndTail(self,singletablecontent):
        #长表格的头尾都要替换，因此单独封装成一个函数，否则长表格的表格需号将加2
        #替换第一行为长表格
        searchstr = r'(\\begin{savenotes}\\sphinxattablestart)'
        splittable = re.search(searchstr, singletablecontent,re.M | re.I|re.U )
        #替换为长表格
        tablefirstline = re.sub(r'\\sphinxattablestart',r'\\sphinxatlongtablestart', splittable.group(0), re.M|re.I|re.U)
        #替换第2行为长表格
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent,re.I|re.U )
        #记录表头的位置
        headlastpos = splittable.end()
        tablesecondline = re.sub(r'\\begin{tabular}|\\begin{tabulary}',r'\\begin{longtable}', splittable.group(0), re.I|re.U)
        tablesecondline = re.sub(r'\{\\linewidth\}',r'', tablesecondline, re.I|re.U)

        #查找caption
        searchstr = r'\\sphinxcaption([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent, re.I|re.U)
        longcaption = re.sub(r'\\sphinxcaption',r'\\caption', splittable.group(0), re.I|re.U)
        #添加长表格专用指令
        longcaption += r"\\*[\sphinxlongtablecapskipadjust]"
        
        #构建长表个的表头部分
        longhead = tablefirstline + tablesecondline + '\n' + r'\sphinxthelongtablecaptionisattop' + '\n'+ longcaption+'\n'
        
        #替换表尾
        newtablecontent = singletablecontent[headlastpos:len(singletablecontent)]
        endtable = re.sub(r'(\\end{tabular}|\\end{tabulary})',r'\\end{longtable}', newtablecontent, re.M | re.I|re.U)
        endtable = re.sub(r'\\par',r'', endtable, re.M | re.I|re.U)
        endtable = re.sub(r'\\sphinxattableend',r'\\sphinxatlongtableend', endtable, re.M | re.I|re.U)
        
        #生成新的长表格内容
        singletablecontent = longhead + endtable
        return singletablecontent
        
    #为表格增加h属性对齐方式，避免表格浮动，让表格在当前位置显示    
    def __AddHAttrFroTable(self,content):
    
        searchstr = r'(\\begin{tabular}|\\begin{tabulary}|\\begin{longtable})({\\linewidth})?(\[(?P<attr>[a-z]{1,4})\])?([\s\S].*)'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        attrcontent = m.group('attr')
        posarr = m.span()
        if m.group(3) == '':   #group(3)是[htp]的组合，如果没有表格属性则添加上表格属性
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+r'[h]'+m.group(5)+content[posarr[1]:len(content)]
        else:
            replacestr = m.group(4)  #需要被替换的表格属性
            #判断该表格是否有P属性，如果有p属性需要删掉
            replacestr.replace('p','') 
            #给该表格添加h属性，避免表格浮动，让表格在当前位置显示
            replacestr = 'h'+replacestr
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+'['+replacestr+']'+m.group(5)+content[posarr[1]:len(content)]
        #print('newcontent: %s' % newcontent)
        return newcontent
        
    #修改sphinx自动生成的长表格表头
    def __ModifyLongTableHead(self,content,headtype):

        #先找出第一行
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(content)
        posarr = []
        i = 0
        for m in matchiter:
            
            if i > 1:
                break;
                
            posarr.append([])
            posarr[i] = m.span() #保存起始位置和结束位置，便于组成新的内容
            
            if i ==0:
                newcontent = content[0:posarr[i][0]]
            else:
                newcontent = newcontent+content[posarr[i-1][1]:posarr[i][0]]
                
            newcontent += r'\hline\rowcolor'+headtype
            
            headcontent = m.group(1) #匹配到的第一个即为表头内容
            
            if 'multicolumn' in headcontent:
                return content        
            
            headlist = []
            
            if r'\sphinxstyletheadfamily' in headcontent:
                pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
                aftercontent = headcontent
                mobjarr = pattern.finditer(aftercontent)
                
                preposlist = []
                for mobj in mobjarr:
                    amarr = mobj.group('value')
                    curposlist = mobj.span()
            
                    #用表头内容数组替换
                    fontcolor = self.tablesattrobj.headfontcolor
                    #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
                    amarr = amarr.strip()
                    amarr = amarr.strip('\r')
                    amarr = amarr.strip('\n')
                    amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
                    if amarr == '':
                        continue
                    fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
                    if len(preposlist) > 0:
                        headlist.append(headcontent[preposlist[1]:curposlist[0]])
                    else:
                        headlist.append(headcontent[0:curposlist[0]])
                    headlist.append(fontcolor)
                    preposlist = curposlist
                headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
                headcontent = ''
                for prelist in headlist:
                    headcontent = headcontent + prelist + '\n'
                newcontent += headcontent+r'\hline'               
            i +=1
        newcontent += content[posarr[i-1][1]:len(content)]
        return newcontent

    def __ModifyTableHead(self, content, headtype):
        #先找出第一行
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        headcontent = m.group(1) #匹配到的第一个即为表头内容
        posarr = m.span(1)  #保存起始位置和结束位置，便于组成新的内容

        if 'multicolumn' in headcontent:
            return content        

        if r'\sphinxstyletheadfamily' in headcontent:
            pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
            aftercontent = headcontent
            #pattern = re.compile(r'(?<=\\sphinxstylethead{\\sphinxstyletheadfamily)([\s\S]*?)(?=\\unskip}\\relax &)', re.M | re.I|re.U)
        else:
            aftercontent = headcontent.replace(r'\\','&',1)
            pattern = re.compile(r'(?P<value>[\s\S]*?)(&\s{1})', re.M | re.I|re.U)
            #pattern = re.compile(r'[\s\S]*?&|\\unskip', re.M | re.I|re.U)

        mobjarr = pattern.finditer(aftercontent)
        headlist = []
        preposlist = []
        for mobj in mobjarr:
            amarr = mobj.group('value')
            curposlist = [mobj.start(),mobj.start()+len(amarr)]

            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            if amarr == '':
                continue
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            if len(preposlist) > 0:
                headlist.append(headcontent[preposlist[1]:curposlist[0]])
            else:
                headlist.append(headcontent[0:curposlist[0]])
            headlist.append(fontcolor)
            preposlist = curposlist

        headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
        headcontent = ''
        for prelist in headlist:
            headcontent = headcontent + prelist + '\n'
        newcontent = content[0:posarr[0]]+r'\rowcolor'+headtype+'\n'+headcontent+content[posarr[1]:len(content)]
        return newcontent
    #渲染树型表格的第一列    
    def __ModifyVerticalTable(self, singletablecontent):
        #找出每一行的内容
        searchstr = r'(?<=\\hline)(?P<content>[\s\S]*?)(?=\\hline)'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(singletablecontent)
        
        posarr=[]  #保存位置，便于组合
        i = 0
        for m in matchiter:
        
            posarr.append([])
            posarr[i] = m.span()
            
            if i ==0:
                newcontent = singletablecontent[0:posarr[i][0]]
            else:
                newcontent = newcontent+singletablecontent[posarr[i-1][1]:posarr[i][0]]
        
            cellcontent = m.group('content') #匹配到的第一个即为表头内容
            #将第一个单元格内容渲染成蓝底白字
            firstcellcontent = self.__ModifyFirstColumnType(cellcontent)
            newcontent += firstcellcontent 
            i+=1
        newcontent += singletablecontent[posarr[i-1][1]:len(singletablecontent)]
        return newcontent
            
    #渲染第一个单元格内容
    def __ModifyFirstColumnType(self,cellcontent):
    
        new_cellcontent = ""

        if r'\sphinxstyletheadfamily' in cellcontent:
        
            searchstr = r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)'
            
            aftercontent = cellcontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()
            mobj = re.search(searchstr, aftercontent, re.M|re.I|re.U )  #匹配到的第一个既是需要修改的内容
            
            #修改字体颜色
            amarr = mobj.group('value')
            posarr = mobj.span()
            new_cellcontent = aftercontent[0:posarr[0]]+'\n'+r'\cellcolor'+self.tablesattrobj.headtype
            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            #if amarr == '':
            #    continue
            if (r'\textbf' or r'\textcolor') in amarr:
                return cellcontent
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            new_cellcontent +=r'{'+fontcolor + '}\n' + aftercontent[posarr[1]:len(aftercontent)]
        else:
            
            aftercontent = cellcontent.replace(r'\\','&',1)
            #去掉首尾空格和换行符
            aftercontent = aftercontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()
            
            tmplist = re.split(r'&',aftercontent)

            preposlist = 0
            #只对第一个做修改
            onelist = tmplist[0]
            
            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            onelist = onelist.strip()
            onelist = onelist.strip('\r')
            onelist = onelist.strip('\n')
            onelist = onelist.strip()  #再去掉首尾空格，避免多余的空格出现
            #if onelist == '':
            #        continue
            if (r'\textbf' or r'\textcolor') in onelist:
                return cellcontent
            new_cellcontent = '\n'+r'\cellcolor'+self.tablesattrobj.headtype+r'{'+fontcolor.replace('{}','{'+ onelist+'}',1)+r'}'+'\n'

            for i in range(1,len(tmplist)):
                if len(tmplist[i])>0:
                    new_cellcontent += '&' +tmplist[i]
            new_cellcontent+=r'\\' #将最后被替换掉的\\再加上

        return new_cellcontent + '\n'

# 打开Makefile文件查找source和build文件夹
def OpenMakefile():
    global source_dir
    global build_dir
    source_dir = ''
    build_dir = ''
    try:
        if platform.system().lower() == 'windows':
            with open('make.bat',"r") as f:
                fstr = f.read()
                
            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"set *SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为source所在目录
            
            searchstr = r"set *BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为build所在目录
        else:
            with open('Makefile',"r") as f:
                fstr = f.read()
            
            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为源所在目录
            
            searchstr = r"BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为源所在目录
        
    except Exception as e:
        print(e)
        return

def GetLatex_documents():
    global source_dir
    if source_dir == '':
        return
    #得到配置文件conf.py的路径
    if source_dir == '.':
        confdir = './conf.py'
    else:
        confdir = './' + source_dir +'/conf.py'

    conffile = os.path.abspath(confdir)
     #打开conf.py文件
    with codecs.open(conffile,"r+",encoding='utf-8') as f:
            fstr = f.read()
    list = []
    versioninfo = GetVersionInfo(fstr)
    fileprefile = GetFilePreInfo(fstr)
    if versioninfo=="" or fileprefile=="":
    #根据正则表达式，找出latex_documents内容
        searchstr = r"latex_documents *= *\[([\s\S]*?)\]"
        m = re.search(searchstr, fstr, re.M|re.I|re.U )
        latex_documents = m.group(1) #匹配到的第一个即为源所在目录
        #拆分二维数组，兼容多个情况
        list = latex_documents.split(")")
        for i in range(len(list)):
            if IsComment(list[i]):
                list[i]= list[i].split(",")
        list.pop()
    else:
        #创建2维列表
        for i in range(1):
            list.append([])
            for j in range(2):
                list[i].append("")
        list[0][0] = "comment"  #为了兼容文件解析内容，进行补位。
        list[0][1] = '"' + fileprefile + versioninfo + '.tex"'  #为了兼容解析过程，添加双引号。
    return list

#得到版本信息
def GetVersionInfo(fstr):

    if releasever != '':
       return releasever
       
    versioninfo = ""
    searchstr = r"version *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.I)
    if m != None:
        versioninfo = m.group(2) #匹配到的第一个即为源所在目录
    print("version = " + versioninfo)
    return versioninfo

#得到文件前缀信息
def GetFilePreInfo(fstr):
    filepre = ""
    searchstr = r"curfnpre *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.M|re.I|re.U )
    if m != None:
        filepre = m.group(2) #匹配到的第一个即为源所在目录
    print("filepre = " + filepre)
    return filepre

#判断是否为注释行
def IsComment(instr):

    if instr.strip() is None:
        return False

    rule = re.compile('^#.*$')
    if rule.match(instr.strip()) is None:
        return True;
    else:
        return False;

#根据正则表达式取单引号和双引号中的内容
def getquomarkcontent(strarr):
    #根据正则表达式，找出双引号和单引号中的内容
    print("texfile="+strarr)
    searchstr = r"[\"|'](.*?)[\"|']"
    m = re.search(searchstr, strarr, re.M|re.I|re.U )
    if m is None:
        return None
    return m.group(1).strip() #匹配到的第一个即为源所在目录

def Modifylatex_main(build_dir,latexdocumentslst):
    
    global __load_dict__
    
    #__load_dict__ = __openconfjsonfile__()
    
    #print('parsejson=' + build_dir)
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(latexdocumentslst)
    
    doclen = len(latexdocumentslst)
    for i in range(0,doclen):
        #得到latex路径
        latexpath = build_dir
        #copy 背景图到latex路径，背景图必须与该文件在同一个目录下
        if os.path.exists('./chapterbkpaper.pdf'):
            shutil.copy('./chapterbkpaper.pdf',latexpath)
        #得到相对路径
        filename = latexdocumentslst[i][1]
        #print('filename='+filename)
        if  filename is None:
            continue
        texfilepath = latexpath + '/' + filename
        #相对路径转绝对路径
        texfile = os.path.abspath(texfilepath)
        if not os.path.exists(texfile):
            continue
        #print('texfile=' + texfile)
        fo = codecs.open(texfile, "r+",encoding = 'utf-8')
        texcontent = fo.read()
        fo.close
        
        #得到修改tex文件的对象
        ModTexobj = clsModifyTex(texcontent)
        ModTexobj.AddPackageToTex()
        ModTexobj.AddOtherTocToTex()
        ModTexobj.AddCustormOptionsToTex()
        ModTexobj.ModifyReplacePackage()
        ModTexobj.ModifyFunctionBackColor()
        ModTexobj.ModifyTablesAttributes()
        
        fw = codecs.open(texfile, "w+",encoding = 'utf-8')
        fw.write(ModTexobj.content)
        fw.close 
def parseargv():
    global source_dir
    global latex_dir
    if len(sys.argv) <= 1:
        return
    for i in range(1,len(sys.argv)):
        param = sys.argv[i]
        if param=='-c':  #解析conf.py所在目录
            source_dir = sys.argv[i+1] #保存相对路径
            print("argv source=" + source_dir)
            i = i+2
        elif param=='-l': #保存输出的latex目录
            latex_dir = sys.argv[i+1] #保存相对路径
            i = i+2
            print("latex_dir = "+latex_dir)
            
source_dir = '' #保存源文件所在目录
build_dir = ''  #保存生成文件所在目录
releasever = ''
latex_dir = ''
__load_dict__ = __openconfjsonfile__()

if __name__ == '__main__':

    parseargv()  #解析系统参数
    OpenMakefile()
        
    latex_documents = GetLatex_documents() #保存latex文档所在数组
    #__load_dict__ = __openconfjsonfile__()
    
    latexdir=""
    if latex_dir == '':
        latexdir = '/latex/'
    else:
        latexdir = '/' + latex_dir + '/'
        
    doclen = len(latex_documents)
    for i in range(0,doclen):
        #得到latex路径
        latexpath = './' + build_dir + latexdir
        #copy 背景图到latex路径，背景图必须与该文件在同一个目录下
        if os.path.exists('./chapterbkpaper.pdf'):
            shutil.copy('./chapterbkpaper.pdf',latexpath)
        #得到相对路径
        if getquomarkcontent(latex_documents[i][1]) is None:
            continue
        texfilepath = latexpath + getquomarkcontent(latex_documents[i][1])
        #相对路径转绝对路径
        texfile = os.path.abspath(texfilepath)
        if not os.path.exists(texfile):
            continue
        fo = codecs.open(texfile, "r+",encoding = 'utf-8')
        texcontent = fo.read()
        fo.close
        
        #得到修改tex文件的对象
        ModTexobj = clsModifyTex(texcontent)
        ModTexobj.AddPackageToTex()
        ModTexobj.AddOtherTocToTex()
        ModTexobj.AddCustormOptionsToTex()
        ModTexobj.ModifyReplacePackage()
        ModTexobj.ModifyFunctionBackColor()
        ModTexobj.ModifyTablesAttributes()
        
        fw = codecs.open(texfile, "w+",encoding = 'utf-8')
        fw.write(ModTexobj.content)
        fw.close



