from __future__ import print_function
import os

def conllWrite(outputPath, sentences, headers):
    """
    Writes a sentences array/hashmap to a CoNLL format
    """
    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))
    fOut = open(outputPath, 'w')
    
    
    for sentence in sentences:
        fOut.write("#")
        fOut.write("\t".join(headers))
        fOut.write("\n")
        for tokenIdx in range(len(sentence[headers[0]])):
            aceData = [sentence[key][tokenIdx] for key in headers]
            fOut.write("\t".join(aceData))
            fOut.write("\n")
        fOut.write("\n")


def checkAndCorrectBIOEncoding(predictions):
    """ Check and convert inplace wrong BIO encoding to correct BIO encoding """ 
    errors = 0
    labels = 0
    
    for sentenceIdx in range(len(predictions)):
        labelStarted = False
        labelClass = None
        

        for labelIdx in range(len(predictions[sentenceIdx])): 
            label = predictions[sentenceIdx][labelIdx]
            labelNext = predictions[sentenceIdx][labelIdx+1] if labelIdx < len(predictions[sentenceIdx]) - 1 else 'O'
            labelPrev = predictions[sentenceIdx][labelIdx-1] if labelIdx > 0 else 'O'
            labelClass = label[2:] if len(label) > 1 else label
            labelClassNext = labelNext[2:] if len(labelNext) > 1 else labelNext
            labelClassPrev = labelPrev[2:] if len(labelPrev) > 1 else labelPrev
            if label != 'O':
                if label.startswith('I-') and labelClassPrev != labelClass:
                     errors += 1
                     predictions[sentenceIdx][labelIdx] = 'B'+predictions[sentenceIdx][labelIdx][1:]    
    if errors > 0:
        labels += errors
        logging.info("Wrong BIO-Encoding %d/%d labels when setting incorrect labels" % (errors, labels))

        
        
def readCoNLL(inputPath, cols, commentSymbol=None):
    """
    Reads in a CoNLL file and returns a list with sentences (each sentence is a list of tokens)
    """
    sentences = []
    
    sentenceTemplate = {name: [] for name in cols.values()}
    
    sentence = {name: [] for name in sentenceTemplate.keys()}
    
    newData = False
    lastval = 'O'    
    for line in open(inputPath):
        line = line.strip().replace('\xa0','_')
        if len(line) == 0 or (commentSymbol != None and line.startswith(commentSymbol)):
            if newData:      
                sentences.append(sentence)
                    
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue
        
        splits = line.split()
        print(splits)

        for colIdx, colName in cols.items():
            val = splits[colIdx]
            sentence[colName].append(val)

 
        newData = True  
        
    if newData:        
        sentences.append(sentence)
    
            
    for name in cols.values():
        if name.endswith('_IOB'):

            #FIX using IOB2 or BIO: if 'I-MISC', 'I-MISC', 'O', 'I-PER', 'I-PER', -> converts into -> 'B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER',
            className = name[0:-4]+'_IOB2'
            for sentence in sentences:
                sentence[className] = []
                lastval = 'O'
                for val in sentence[name]:
                    newval = val
                    if (lastval =='O' and val[0]=='I') or (val != 'O' and lastval != 'O' and lastval[1:] != val[1:]):
                        newval='B'+val[1:]
                    lastval = val
                    val = newval
                    sentence[className].append(val)
            
            #Add class
            className = name[0:-4]+'_class'
            for sentence in sentences:
                sentence[className] = []
                for val in sentence[name]:
                    valClass = val[2:] if val != 'O' else 'O'
                    sentence[className].append(valClass)
                    
            #Add IOB encoding
            iobName = name[0:-4]+'_IOBX'
            for sentence in sentences:
                sentence[iobName] = []
                oldVal = 'O'
                for val in sentence[name]:
                    newVal = val
                    
                    if newVal[0] == 'B':
                        if oldVal != 'I'+newVal[1:]:
                            newVal = 'I'+newVal[1:]
                        

                    sentence[iobName].append(newVal)                    
                    oldVal = newVal
                    
            #Add IOBES encoding
            iobesName = name[0:-4]+'_IOBES'
            name = name[0:-4]+'_IOB2'
            for sentence in sentences:
                sentence[iobesName] = []
                
                for pos in range(len(sentence[name])):                    
                    val = sentence[name][pos]
                    nextVal = sentence[name][pos+1] if (pos+1) < len(sentence[name]) else 'O'
                    prevVal = sentence[name][pos-1] if pos > 0 else 'O'
                    
                    
                    newVal = val
                    if val[0] == 'B' and nextVal[0] != 'I':
                        newVal = 'S'+val[1:]
                    elif val[0] == 'I' and  nextVal[0] != 'I':
                        newVal = 'E'+val[1:]
                    sentence[iobesName].append(newVal)                    
                   
    return sentences  



           
        
