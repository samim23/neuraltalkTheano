########### Eval using coco toolkit ##########

resDump = []
for img in resMulti['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resDump.append({'image_id': imgid, 'caption':img['candidate']['text'].lstrip(' ').rstrip(' ')})


#################################
f = open('CandCommitteCocoMert.txt','w')
scrs = evalSc['logProb_feat']
cnt = 0
icnt = 0
for img in resOrig['images']:
    for s in img['sentences']:
        f.writelines(('%d ||| %s ||| %d '%(icnt,s['raw'],len(s['tokens']))) + ' '.join(map(str,scrs[:,cnt])) +'\n')
        cnt += 1
    icnt += 1


mod_names = {}
rootdir = '/ssdscratch/jormal/picsom/databases/COCO/objects/'
for r in os.walk(rootdir):
    if len(r[1]) == 0:
        for fl in r[2]:
            if 'eval' in fl:
               Cands =  


all_references = {}
for img in dataset['images']:
    references = [' '.join(x['tokens']) for x in img['sentences']] # as list of lists of tokens
    all_references[img['cocoid']] = references

trn_refernces = [[] for q in xrange(5)]

for img in trnData['imgblob']:
    for q in xrange(5):
        trn_refernces[q].append(all_references[int(img['imgid'])][q])

for q in xrange(5):
   open('./nlpUtils/zmert_v1.50/zmert_ex_coco/referencefull.'+`q`, 'w').write('\n'.join(trn_refernces[q]))


nnStatsF = open('NNStats/Fast_SearchResult_MertAllModel.txt','r').read().splitlines()
nnStats = []
for ln in nnStatsF:
    if 'NN for tweet' == ln[:12]:
        nnStats.append(float(ln.split('= ')[1]))


import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data
params = {}
params['dataset'] = 'coco'
params['data_file'] = 'dataset.json'
params['feature_file'] = 'ConvAE_Test_e50_f32.hdf5'



Proj = np.random.rand(4096,out['image']['feat'].shape[0])
row_sum = Proj.sum(axis=1)
Proj = Proj/row_sum[:,np.newaxis]


i=0
for i in xrange(totImgs):
    cFeats[:,i] = Proj.dot(dp.features[i,:])
    if i %10 == 1:
        print('%d'%i)


############### Dump result Struct #############################


caps  =[]
for img in resOrig['imgblobs']:
    imgid = int(img['img_path'].rsplit('_')[-1].split('.')[0])
    caps.append({'image_id':imgid, 'caption':img['candidate']['text']})



############### Computing mutual Information #################

import numpy as np
import cPickle as pickle
import json
from operator import itemgetter

checkpoint = pickle.load(open('trainedModels/model_checkpoint_coco_gpu001_c_in14_o9_fc7_d_a_Auxo9_fc8_11.96.p','r'))
wix = checkpoint['wordtoix']
dataset = json.load(open('/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/instances_train2014.json','r'))
ixw = checkpoint['ixtoword']

from collections import defaultdict
from imagernn.data_provider import getDataProvider, prepare_data
params = {}
params['dataset'] = 'coco'
params['data_file'] = 'dataset.json'
dp = getDataProvider(params)


catIdImgs = defaultdict(set)
for ann in dataset['annotations']:
    catIdImgs[ann['category_id']].add(ann['image_id'])

catIdtoIx = {}
for i,cat in enumerate(catIdImgs.keys()):
    catIdtoIx[cat] = i

nTrnSamp = len(dataset['images'])
wordsIdList = defaultdict(set)

for img in dp.split['train']:
    for sent in img['sentences']:
        for tk in sent['tokens']:
            if tk in wix:
                wordsIdList[tk].add(img['cocoid'])

wordProbs = np.zeros(len(wix))
for w in wordsIdList:
    wordProbs[wix[w]] = float(len(wordsIdList[w]))/nTrnSamp

wordProbs[0] = 1

catProbs = np.zeros(len(catIdtoIx))
for i in catIdImgs:
    catProbs[catIdtoIx[i]] = float(len(catIdImgs[i]))/nTrnSamp

mi = np.zeros((len(catIdImgs),len(wix)))
jp = np.zeros((len(catIdImgs),len(wix)))
delt = np.zeros((len(catIdImgs),len(wix)))

totImgs = float(len(dp.split['train']))
eps = 1e-10

for cid in catIdImgs:
    for tk in wordsIdList:
        jp[catIdtoIx[cid],wix[tk]] = float(len(wordsIdList[tk] & catIdImgs[cid])+eps)

sumd = nTrnSamp #np.sum(jp)

for cid in catIdImgs:
    for tk in wordsIdList:
        mind = min(wordProbs[wix[tk]], catProbs[catIdtoIx[cid]])
        delt[catIdtoIx[cid],wix[tk]] = (jp[catIdtoIx[cid],wix[tk]] /(jp[catIdtoIx[cid],wix[tk]] + 1)) * (mind*nTrnSamp/(mind*nTrnSamp+1))

jp = jp/sumd
jp[:,0] = 1.0 + eps


for cid in catIdImgs:
    for tk in wordsIdList:
        mi[catIdtoIx[cid],wix[tk]] = np.log((jp[catIdtoIx[cid],wix[tk]]) / (wordProbs[wix[tk]] * catProbs[catIdtoIx[cid]]))

mi = mi*delt/ -np.log(jp)
ixtocat = {}
for nm in dataset['categories']:
    ixtocat[catIdtoIx[nm['id']]] = nm['name']

#wVecMat = mi
#wVecMat = checkpoint['model']['Wemb'].T
wVecMat = np.concatenate((checkpoint['model']['Wemb'].T,checkpoint['model']['Wd']),axis=0)

normWords = np.sqrt(np.sum(wVecMat**2,axis = 0)[:,np.newaxis]) + 1e-20
wordsMutualSim = wVecMat.T.dot(wVecMat) / normWords.dot(normWords.T)

catToWords = defaultdict(list)
for i in xrange(mi.shape[1]):
    cid = mi[:,i].argmax()
    catToWords[cid].append((i,mi[cid,i]))

k = 10
fid = open('./nlpExpts/results/wordToCatMap/cat2words_Top10_PPmiSmooth_nomrz_OnlyJJ_Corr.txt','w')
for cid in catToWords:
    fid.write('\n%s : '%(ixtocat[cid]))
    srtedK = sorted(catToWords[cid],key=itemgetter(1),reverse=True)[:k]
    widx = [Wc[0] for Wc in  srtedK]
    scrs = mi[cid,widx] 
    newScrs = (scrs/scrs[0] + wordsMutualSim[widx,widx[0]])/ 2
    widx2 = np.argsort( newScrs)[::-1] 
    for w in widx2:
        fid.write('%s (%.2f), '%(ixw[widx[w]],newScrs[w]))
fid.close()

fid = open('./nlpExpts/results/wordToCatMap/word2cat_Top10_PPmiSmooth_OnlyJJ_Corr.txt','w')
for cid in ixtocat:
    fid.write('\n%s : '%(ixtocat[cid]))
    for i in mi[cid,:].argsort()[::-1][:k]:
        fid.write('%s (%.2f), '%(ixw[i],mi[cid,i]))
fid.close()

visWordsDict = defaultdict(set)

fid = open('./nlpExpts/results/wordToCatMap/word2cat_Top10_PPmiSmooth_OnlyJJ_Corr_SimComb.txt','w')
for cid in ixtocat:
    fid.write('\n%s : '%(ixtocat[cid]))
    widx = mi[cid,:].argsort()[::-1][:k]
    newScrs = (mi[cid,widx]/mi[cid,widx[0]] + wordsMutualSim[widx,widx[0]])/ 2
    widx2 = np.argsort(newScrs)[::-1] 
    for i in widx2:
        if newScrs[i] > 0.70:
            visWordsDict[ixw[widx[i]]].add(ixtocat[cid])    
        fid.write('%s (%.2f), '%(ixw[widx[i]],newScrs[i]))
fid.close()
############################################################################################################################################
fid = open('./nlpExpts/data/dbSentencesRaw.txt','w')
fid.write(' .\n'.join([re.sub('\\n+','. ',ann['caption'].lstrip(' .\n').rstrip(' .\n')) for ann in dbTrn['annotations']]) + ' .')
fid.close()

########################################### Visualizing Generated vocabulary ################################################################
dataset = json.load(open('data/coco/dataset.json','r'))
subResultsT = json.load(open('eval/mseval/results/captions_test2014_CMME_results.json','r'))
subResultsV = json.load(open('eval/mseval/results/captions_val2014_CMME_results.json','r'))

nSampT = len(subResultsT)
nSampV = len(subResultsV)

genDictT = defaultdict(int)
for res in subResultsT:
    for w in res['caption'].split(' '):
        genDictT[w] += 1

genCntsT = np.zeros(len(genDictT))
ixwGenT = {}
for i,w in enumerate(genDictT):
    genCntsT[i] = genDictT[w]
    ixwGenT[i] = w


genDictV = defaultdict(int)
for res in subResultsV:
    for w in res['caption'].split(' '):
        genDictV[w] += 1

genCntsV = np.zeros(len(genDictV))
ixwGenV = {}
for i,w in enumerate(genDictV):
    genCntsV[i] = genDictV[w]
    ixwGenV[i] = w



nSamples = 0
trnDict = defaultdict(int)
for img in dataset['images']:
    if img['split'] == 'train':
        for s in img['sentences']:
            for w in s['tokens']:
                trnDict[w] += 1.0
            nSamples += 1

nSampVRef = 0
valDict = defaultdict(int)
for img in dataset['images']:
    if img['split'] != 'train':
        for s in img['sentences']:
            for w in s['tokens']:
                valDict[w] += 1.0
            nSampVRef += 1


srtidx = np.argsort(genCntsT)[::-1]
genwordsSrtedT = [ixwGenT[i] for i in srtidx]
genCntsT = genCntsT[srtidx]/nSampT
trnCntsT = np.array([trnDict[w] for w in genwordsSrtedT]) / nSamples

srtidx = np.argsort(genCntsV)[::-1]
genwordsSrtedV = [ixwGenV[i] for i in srtidx]
genCntsV = genCntsV[srtidx]/nSampV
trnCntsV = np.array([trnDict[w] for w in genwordsSrtedV]) / nSamples
valCntsV = np.array([valDict[w] for w in genwordsSrtedV]) / nSampVRef 



trnCntsAll = np.zeros(len(trnDict))
ixwTrn = {}
for i,w in enumerate(trnDict):
    trnCntsAll[i] = trnDict[w]
    ixwTrn[i] = w
srtidx = np.argsort(trnCntsAll)[::-1][:8900]
trnwordsSrted = [ixwTrn[i] for i in srtidx]
trnCntsAll = trnCntsAll[srtidx]/ nSamples
genCntsTall = np.zeros(trnCntsAll.shape[0]) + 1e-6
genCntsVall = np.zeros(trnCntsAll.shape[0]) + 1e-6
valCntsall = np.zeros(trnCntsAll.shape[0]) + 1e-6

for i,w in enumerate(trnwordsSrted):
    if w in genDictT:
        genCntsTall[i] = float(genDictT[w]) / nSampT
    if w in genDictV:
        genCntsVall[i] = float(genDictV[w]) / nSampV
    if w in valDict:
        valCntsall[i] = float(valDict[w]) / nSampVRef

filepath = 'generateWordsVisTest.html'
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
source1 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(genCntsT),cnt = genCntsT*nSampT,lab = genwordsSrtedT))
p1 = figure(title="WC/sent(log10) in generted (TEST) vs Train", tools=TOOLS)
p1.circle(range(genCntsT.shape[0]),np.log10(genCntsT),source = source1,color="blue",legend = "Cnt in Generated")
source2 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(trnCntsT),cnt=trnCntsT*nSampT,lab = genwordsSrtedT))
p1.circle(range(genCntsT.shape[0]),np.log10(trnCntsT),source = source2,legend = "Cnt in Train",color="red")
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

source3 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(genCntsT/trnCntsT),cnt = genCntsT/trnCntsT,lab = genwordsSrtedT))
p2 = figure(title="Ratio of WC/sent in log TEST vs Train", tools=TOOLS)
p2.square(range(genCntsT.shape[0]),np.log10(genCntsT/trnCntsT),source = source3,color="blue",legend = "Gen Test / Train")
p2.line(range(genCntsT.shape[0]),np.log10(genCntsT/trnCntsT),source = source3,color="blue",legend = "Gen Test / Train")
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(x,ratio)","(@x,@cnt)"),("text","@lab")]


source4 = ColumnDataSource(data=dict(x=range(trnCntsAll.shape[0]), y=np.log10(genCntsTall),cnt = genCntsTall*nSampT,lab = trnwordsSrted))
p3 = figure(title="WC/sent(log10) in generted (TEST) vs Train", tools=TOOLS,plot_width=1200)
p3.circle(range(genCntsTall.shape[0]),np.log10(genCntsTall),source = source4,color="blue",legend = "Cnt in Generated")
p3.line(range(genCntsTall.shape[0]),np.log10(genCntsTall),line_dash=[4, 4],source = source4,color="blue",legend = "Cnt in Generated")
source5 = ColumnDataSource(data=dict(x=range(genCntsTall.shape[0]), y=np.log10(trnCntsAll),cnt=trnCntsAll*nSamples,lab = trnwordsSrted))
p3.circle(range(genCntsTall.shape[0]),np.log10(trnCntsAll),source = source5,legend = "Cnt in Train",color="red")
hover3 = p3.select(dict(type=HoverTool))
hover3.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

save(VBox(HBox(p1,p2),p3))


filepath = 'generateWordsVisVal.html'
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
p1 = figure(title="WC/sent(log10) in generted (Val) vs Train", tools=TOOLS)
##
source1 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(genCntsV),cnt = genCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(genCntsV.shape[0]),np.log10(genCntsV),source = source1,color="blue",legend = "Cnt in Generated")
##
source2 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(trnCntsV),cnt=trnCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(genCntsV.shape[0]),np.log10(trnCntsV),source = source2,fill_alpha = 0.5,legend = "Cnt in Train",color="red")
##
source1V = ColumnDataSource(data=dict(x=range(valCntsV.shape[0]), y=np.log10(valCntsV),cnt=valCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(valCntsV.shape[0]),np.log10(valCntsV),source = source1V,fill_alpha = 0.5,legend = "Cnt in Val Ref",color="black")
##
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

source3 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(genCntsV/trnCntsV),cnt = genCntsV/trnCntsV,lab = genwordsSrtedV))
p2 = figure(title="Ratio of WC/sent in log Val vs Train", tools=TOOLS)
p2.square(range(genCntsV.shape[0]),np.log10(genCntsV/trnCntsV),fill_alpha = 0.5,source = source3,color="blue",legend = "Gen Val / Train")
p2.line(range(genCntsV.shape[0]),np.log10(genCntsV/trnCntsV),line_dash=[4, 4],source = source3,color="blue",legend = "Gen Val / Train")

source3V = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(valCntsV/trnCntsV),cnt = valCntsV/trnCntsV,lab = genwordsSrtedV))
p2.square(range(genCntsV.shape[0]),np.log10(valCntsV/trnCntsV),fill_alpha = 0.5,source = source3V,color="red",legend = "Val Ref / Train")
p2.line(range(genCntsV.shape[0]),np.log10(valCntsV/trnCntsV),line_dash=[4, 4],fill_alpha = 0.3,source = source3V,color="red",legend = "Val Ref / Train")
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(x,ratio)","(@x,@cnt)"),("text","@lab")]


p3 = figure(title="WC/sent(log10) in generted (Val) vs Train", tools=TOOLS,plot_width=1200)
##
source4 = ColumnDataSource(data=dict(x=range(trnCntsAll.shape[0]), y=np.log10(genCntsTall),cnt = genCntsTall*nSampV,lab = trnwordsSrted))
p3.circle(range(genCntsTall.shape[0]),np.log10(genCntsTall),source = source4,color="blue",legend = "Cnt in Generated")
p3.line(range(genCntsTall.shape[0]),np.log10(genCntsTall),line_dash=[4, 4],source = source4,color="blue",legend = "Cnt in Generated")
##
source5 = ColumnDataSource(data=dict(x=range(genCntsTall.shape[0]), y=np.log10(trnCntsAll),cnt=trnCntsAll*nSamples,lab = trnwordsSrted))
p3.circle(range(genCntsTall.shape[0]),np.log10(trnCntsAll),source = source5,fill_alpha = 0.5,legend = "Cnt in Train",color="red")
##
source5V = ColumnDataSource(data=dict(x=range(valCntsall.shape[0]), y=np.log10(valCntsall),cnt=valCntsall*nSampVRef,lab = trnwordsSrted))
p3.circle(range(genCntsTall.shape[0]),np.log10(valCntsall),source = source5V,fill_alpha = 0.3,legend = "Cnt in Val Reference",color="black")
##
hover3 = p3.select(dict(type=HoverTool))
hover3.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

save(VBox(HBox(p1,p2),p3))




########################

srcWrds = [ixw[i] for i in ixw]
len(srcWrds)
genCntsTall = np.zeros(len(srcWrds))
for i,w in enumerate(srcWrds):
    if w in genDictT:
        genCntsTall[i] = float(genDictT[w])

trnCntsAll = np.zeros(len(srcWrds))
for i,w in enumerate(srcWrds):
    if w in trnDict:
        trnCntsAll[i] = float(trnDict[w])
trnCntsAll[0] = nSamples

params['fname_append'] = 'c_in14_o9_fc7_d_a_Aux3gr_2o_an012_10.51_valSet'

colormap = (np.log10(np.log10(trnCntsAll)) - np.min(np.log10(np.log10(trnCntsAll))))*126 + 1
colorsL = ["#%02x%02x%02x" % (2*t,128-t,250*g ) for  t,g in zip(np.floor(colormap),np.floor(genCntsTall>0))]
colors = ["blue","red"]
colorsL2 = [colors[tf] for tf in genCntsTall>0]
radiiGen = (np.log10(genCntsTall/nSampT+1e-6) + 7 )*2 + 3

filepath = 'scatter_Wemb_Callback%s.html' % (params['fname_append'] ) 
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"

radiiGenLcl = radiiGen.copy()

p1 = figure(title="Word embedding Matrix rows", tools=TOOLS,plot_width=1200,plot_height = 900)
source1 = ColumnDataSource(data=dict(x=z1[:,0], y=z1[:,1],cntOrig = trnCntsAll, cntGen = genCntsTall, r = radiiGenLcl, lab = srcWrds))
p1.circle('x', 'y',size = 'r',fill_alpha = (0.8 - 0.5*(genCntsTall > 0)),color = colorsL2, source=source1,line_color=None)
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(cT,cG)","(@cntOrig,@cntGen)"),("text","@lab")]

p2 = figure(title="Word decoding Matrix rows", tools=TOOLS,plot_width=1200,plot_height = 900)
source2 = ColumnDataSource(data=dict(x=z2[:,0], y=z2[:,1],cntOrig = trnCntsAll, cntGen = genCntsTall, r = radiiGenLcl,lab = srcWrds))
p2.circle('x', 'y',size = 'r',fill_alpha = (0.8 - 0.5*(genCntsTall > 0)),fill_color = colorsL,line_color=line_colors, source=source2)
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(cT,cG)","(@cntOrig,@cntGen)"),("text","@lab")]

callback = Callback(args=dict(source=source2), code="""
    var data = source.get('data');
    var f = cb_obj.get('value')
    r = data['r']
    x = data['x']
    y = data['y']
    wrds = data['lab']
    for (i = 0; i < wrds.length; i++) {
        if (f == wrds[i]){          
            r[i] = 30
            break
        }
    }
    source.trigger('change');
""")
text = TextInput(title="Word", name='search', value='one',callback=callback)
save(VBox(text,p2,p1))




