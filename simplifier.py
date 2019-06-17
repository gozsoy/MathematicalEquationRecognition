import cv2
import numpy as np
import pytesseract as pyt
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib
import enchant
import os
import azureTrial

d = enchant.Dict("en_US")


########    HELPERS    ########

# this helper function finds the text segments in 'frame' and eleminates. in the equation, there might be some
# words, and letters, so we should not delete them. these words are passed to function in 'elementsFromLatex'.
# the text finder here is tesseract OCR.
def eleminateText(frame,elementsFromLatex):

    y5=pyt.image_to_data(frame, config='--psm 12')
    if len(y5) != 0:
        y5 = y5.split('\n')
    y5=y5[1:]

    # for each row words found
    for jj in range(len(y5)):

        # split the row into single words
        temp_row=y5[jj].split('\t')

        # words might have below signs before or after. having these signs prevent them to be a valid english word
        #but they actually are. So if a word has one of these before or after, we try that word without these signs
        # to see if it is valid. If so, then eleminate the word.
        accepted=['!','?',',','“','.','”','(',')']

        if len(temp_row[len(temp_row)-1])>1: # This 0 indicates minimum length that will be omitted from deleting. For example, if it is 2, then words with two letters and one letter will not be deleted

            inLatexFlag=False
            temp_word = temp_row[len(temp_row) - 1]

            # if the word is in latex expression coming from handwritten note, we cannot eleminate it.
            if temp_word in elementsFromLatex:
                inLatexFlag=True

            # for detailed explanation, look at line 32
            if not(temp_word in accepted):
                if temp_word[0] in accepted:
                    temp_word = temp_word[1:]

                if temp_word[len(temp_word) - 1] in accepted:
                    temp_word = temp_word[:len(temp_word) - 1]

            # sometimes, tesseract finds actual = as :, but in that case we dont want to eleminate :, since it
            # is actually a useful symbol in our equation. line 56-65 controls this.
            if temp_word!=':':
                if temp_word[0]==':':
                    temp_word = temp_word[1:]
                    if temp_word in elementsFromLatex:
                        inLatexFlag = True

                if temp_word[len(temp_word) - 1]==':':
                    temp_word = temp_word[:len(temp_word) - 1]
                    if temp_word in elementsFromLatex:
                        inLatexFlag = True

            # if the word is not in latex expression, and if it is a meaningful english word, then eleminate it
            # by making the pixels in the vicinity of that word equal to 255(white) or 0(black).
            if inLatexFlag==False:
                if d.check(temp_word):
                    if int(temp_row[len(temp_row)-2])>=70:
                        # below line finds the bounding box around the rectangle
                        x,y,w,h=int(temp_row[len(temp_row)-6]),int(temp_row[len(temp_row)-5]),int(temp_row[len(temp_row)-4]),int(temp_row[len(temp_row)-3])

                        frame[y:y + h, x:x + w]=255

    # returning the text eleminated frame, which is not perfect everytime.
    return frame


# this helper function eleminates the graph area in given 'frame'. it does this job by calling azureTrial.py file.
# .getJSON() function in that file calls Azure API and finds the graph areas, and their probabilities.
def eleminateGraph(frame,image_name):

    prediction_data=azureTrial.getJSON(image_name)

    for iter in range(len(prediction_data)):

        temp_data = prediction_data[iter]

        if temp_data['tagName'] == 'graph' and temp_data['probability'] > 0.7: # this 0.7 is equal to playing with prob treshold in azure website
            x, y, w, h = int(temp_data['boundingBox']['left'] * col), int(temp_data['boundingBox']['top'] * row),int(temp_data['boundingBox']['width'] * col), int(temp_data['boundingBox']['height'] * row)
            #just zero out the bounding box of graph area
            frame[y:y + h, x:x + w] = 0

    return frame

# insert each of the found tesseract result into one big s_and_c array to collect them in a one place.
def insertSymbolData(s_and_c,cr1):
    for i1 in range(len(cr1)):
        temp_symbol=cr1[i1].split(" ")
        if temp_symbol[0]==':':
            temp_symbol[0]='='
        temp_symbol=" ".join(temp_symbol)

        flag=False
        for i11 in range(len(s_and_c)):
            if s_and_c[i11][0]==temp_symbol:
                s_and_c[i11][1]+=1
                flag=True
                break

        if flag != True:
            s_and_c.append([temp_symbol,1])

    return s_and_c

# since each particular tesseract run might be noisy, at least we should take mod of them to get more robust solution.
# here if any of two tesseract runs out of 4 found the same symbol, then I accept it as symbol that really exists in frame.
def get_commons(cr1,cr2,cr3,cr4):
    symbols_and_counts=[]
    morethanonesymbol=[]

    symbols_and_counts = insertSymbolData(symbols_and_counts, cr1)
    symbols_and_counts = insertSymbolData(symbols_and_counts, cr2)
    symbols_and_counts = insertSymbolData(symbols_and_counts, cr3)
    symbols_and_counts = insertSymbolData(symbols_and_counts, cr4)

    print(symbols_and_counts)
    for y1 in range(len(symbols_and_counts)):
        if symbols_and_counts[y1][0].split(" ")[0]=='=':
            morethanonesymbol.append(symbols_and_counts[y1][0])

        elif symbols_and_counts[y1][1]>1:
            morethanonesymbol.append(symbols_and_counts[y1][0])


    return morethanonesymbol

# this helper function is for drawing the rectangle around the found critical symbols and converting all the other frame
# content to black or white pixels.
def drawRectangles(mtos,frame):

    row, col = np.shape(frame)
    temp_frame=np.ones((row,col),int)

    for u in range(len(mtos)):

        temp_symbol=mtos[u].split(" ")
        tempx,tempy,tempw,temph=int(temp_symbol[1]),row-int(temp_symbol[2]),int(temp_symbol[3])-int(temp_symbol[1]),int(temp_symbol[4])-int(temp_symbol[2])

        tempy=tempy-temph
        # the below line is about the rectangle around each critical symbol. you can change it.
        equation_box = [tempx - int(np.floor(col/4)), tempy - int(np.floor(row/6)), int(np.floor(col/2)) +tempw, int(np.floor(row/3)) + temph]

        x, y, w, h = equation_box[0], equation_box[1], equation_box[2], equation_box[3]

        for y_index in range(row):
            for x_index in range(col):
                if y_index>=y and y_index<y+h and x_index>=x and x_index<x+w:
                    temp_frame[y_index, x_index]=frame[y_index, x_index]

    return temp_frame


# find the critical_symbols in the given frame. This symbols should be the ones that are found in the latex expression.
# here I run 4 different tesseract configurations to not miss any critical symbol.
def tesseractRun(frame,critical_symbols):

    y1 = pyt.image_to_boxes(frame, config='--psm 12')
    print(y1)
    if len(y1) != 0:
        y1 = y1.split('\n')
    print("\n")

    y2 = pyt.image_to_boxes(frame, config='--psm 6')
    print(y2)
    if len(y2) != 0:
        y2 = y2.split('\n')
    print("\n")

    y3 = pyt.image_to_boxes(frame, config='--psm 11')
    print(y3)
    if len(y3) != 0:
        y3 = y3.split('\n')
    print("\n")

    y4 = pyt.image_to_boxes(frame)
    print(y4)
    if len(y4) != 0:
        y4 = y4.split('\n')
    print("\n")

    # each critical row is the result of each different tesseract run
    critical_rows1 = []
    critical_rows2 = []
    critical_rows3 = []
    critical_rows4 = []

    for i1 in range(len(y1)):
        if y1[i1][0] == critical_symbols[0] or y1[i1][0] == critical_symbols[1] or y1[i1][0] == critical_symbols[2] or \
                y1[i1][0] == critical_symbols[3]:
            critical_rows1.append(y1[i1])

    for i2 in range(len(y2)):
        if y2[i2][0] == critical_symbols[0] or y2[i2][0] == critical_symbols[1] or y2[i2][0] == critical_symbols[2] or \
                y2[i2][0] == critical_symbols[3]:
            critical_rows2.append(y2[i2])

    for i3 in range(len(y3)):
        if y3[i3][0] == critical_symbols[0] or y3[i3][0] == critical_symbols[1] or y3[i3][0] == critical_symbols[2] or \
                y3[i3][0] == critical_symbols[3]:
            critical_rows3.append(y3[i3])

    for i4 in range(len(y4)):
        if y4[i4][0] == critical_symbols[0] or y4[i4][0] == critical_symbols[1] or y4[i4][0] == critical_symbols[2] or \
                y4[i4][0] == critical_symbols[3]:
            critical_rows4.append(y4[i4])

    return critical_rows1, critical_rows2, critical_rows3, critical_rows4


########    HELPERS END    ########


#######     MAIN STARTS HERE    ######

image_name='cocuk4.png'

frame = cv2.imread(image_name,0)

row,col=np.shape(frame)

frame=eleminateGraph(frame,image_name)

elementsFromLatex=['R','1','2','V','G','b','t','y','z','Z','c','C','Thevenin','R1','reference']

frame=eleminateText(frame,elementsFromLatex)

critical_symbols = ['=', '+', ':', ':']

cr1,cr2,cr3,cr4=tesseractRun(frame,critical_symbols)

morethanonesymbol=get_commons(cr1,cr2,cr3,cr4)

frame=drawRectangles(morethanonesymbol,frame)

imgplot = plt.imshow(frame,cmap='gray')

plt.show()
