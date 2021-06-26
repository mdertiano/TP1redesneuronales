
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def param(matriz_conf,clase,imprimir=True):

  '''
  Esta función calcula los TP,TN,FP,FN para una clase
  -----------
  matriz_conf:   matriz de confusión en numpy array
  clase:         Valor de la clase (posición dentro de la matriz de confusión), si es multiclase se hace one vs all.
  imprimir:      Default True. Imprimir los valores de las métricas
  
  Return:
  -------
  TP,TN,FP,FN
  '''

  TP = matriz_conf[clase,clase]
  FN = np.sum(matriz_conf[clase,:]) - TP
  FP = np.sum(matriz_conf[:,clase])-TP
  TN = np.sum(matriz_conf) - (FP + FN + TP)
  accuracy = (TP)/(np.sum([TP,TN,FP,FN]))
  precision = TP/(TP+FP)
  sensibilidad = TP/(TP+FN)
  f1 = 2*(precision*sensibilidad)/(precision+sensibilidad)
  if imprimir:
    print('One-vs-All, clase nro: '+str(clase))
    print(' ')
    print('TP : ' + str(TP))
    print('TN : ' + str(TN))
    print('FP : ' + str(FP))
    print('FN : ' + str(FN))
    print(' ')
    print('Precision : ' + str(precision))
    print('Sensibilidad : ' + str(sensibilidad))
    print('F1-score : ' + str(f1))

  return TP,TN,FP,FN

#se hace el micro average para sensibilidad y precision -> promedia los valores de sens y precison para cada clase
nro_clases = 20

#nro_clases = 6
def macroAvg(matriz_conf,imprimir=True):
  '''
  Esta función devuelve el macro average de las metricas precisión, sensibilidad y F1-Score
  Argumentos:
  -----------
  matriz_conf:   matriz de confusión en numpy array
  imprimir:      Default True. Imprimir los valores de las métricas
  
  Return:
  -------
  Precision,Sensibilidad, F1-Score
  '''
  P = 0
  S = 0
  #plista_aram: lista de TP,TN,FP,FN
  for i in range(0,nro_clases):
    TP,TN,FP,FN = param(matriz_conf,i,0)
    P = P + (TP/(TP+FP))
    S = S + (TP/(TP+FN))
  P = P/nro_clases
  S = S/nro_clases
  F = 2*(P*S)/(P+S)
  if imprimir:
    print('MacroAvg(presision): ' + str(P))
    print('MacroAvg(sensibilidad): ' + str(S))
    print('MacroAvg(f1-score): ' + str(F))
  return P,S,F
  
def microAvg(matriz_conf,imprimir = True):
  '''
  Esta función devuelve el micro average de las metricas precisión, sensibilidad y F1-Score
  Argumentos:
  -----------
  matriz_conf:   matriz de confusión en numpy array
  imprimir:      Default True. Imprimir los valores de las métricas
  
  Return:
  -------
  Precision,Sensibilidad, F1-Score
  '''
  TP_final = 0
  FP_final = 0
  FN_final = 0
  for i in range(0,nro_clases):
    TP,TN,FP,FN = param(matriz_conf,i,0)
    TP_final = TP_final + TP
    FP_final = FP_final + FP
    FN_final = FN_final + FN
  P = TP_final/(TP_final+FP_final)
  S = (TP_final)/(TP_final+FN_final)
  F = 2*(P*S)/(P+S)
  if imprimir:
    print('MicroAvg(presision): ' + str(P))
    print('MicroAvg(sensibilidad): ' + str(S))
    print('MicroAvg(f1-score): ' + str(F))
  return P,S,F

def acc(matriz_conf, nro_datos):

  '''
  Esta función calcula el Accuracy a partir de una matriz de confusión
  Argumentos:
  -----------
  matriz_conf:   matriz de confusión en numpy array
  nro_datos:     cantidad de datos con los cuales se realiza la matriz de confusión
  
  Return:
  -------
  Accuracy
  '''
  r = np.sum(np.diag(matriz_conf))/nro_datos
  return r

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)